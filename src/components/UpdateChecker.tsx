// Updater UI : verifie les nouvelles releases GitHub au demarrage et offre
// l'install. Alimente par tauri-plugin-updater qui consulte l'endpoint defini
// dans tauri.conf.json (latest.json genere par le workflow CI release).
//
// Comportement :
// - Check au mount (une fois par session).
// - Si une release plus recente existe : bandeau discret en haut avec bouton
//   'Installer'. Le user peut Dismiss pour masquer pendant la session.
// - Progress du download stream via le callback contentLength / downloaded.
// - Une fois downloaded : relaunch() pour redemarrer sur la nouvelle version.
// - Builds GPU (DirectML de la CI, CUDA compile localement) : latest.json ne
//   pointe que vers l'installeur CPU, l'installer en place ferait perdre
//   l'acceleration. Le bouton ouvre donc la page de la release pour que
//   l'utilisateur telecharge la meme variante.

import { useEffect, useState, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { Download, RefreshCw, X } from "lucide-react";
import { listen } from "@tauri-apps/api/event";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import { openUrl } from "@tauri-apps/plugin-opener";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/tauri";
import { cn } from "@/lib/utils";

const RELEASES_URL = "https://github.com/LitteRabbit-37/Parla/releases";

type Status = "idle" | "checking" | "available" | "downloading" | "installed";

export function UpdateChecker() {
  const { t } = useTranslation();
  const [update, setUpdate] = useState<Update | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState<{
    downloaded: number;
    total: number;
  } | null>(null);
  const [dismissed, setDismissed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gpuBuild, setGpuBuild] = useState(false);

  useEffect(() => {
    api
      .parakeetExecutionProvider()
      .then((ep) => setGpuBuild(ep !== "cpu"))
      .catch(() => setGpuBuild(false));
  }, []);

  const runCheck = useCallback(async () => {
    try {
      setStatus("checking");
      setDismissed(false);
      const u = await check();
      if (u) {
        setUpdate(u);
        setStatus("available");
      } else {
        setStatus("idle");
      }
    } catch (e) {
      console.warn("updater check:", e);
      setStatus("idle");
    }
  }, []);

  useEffect(() => {
    runCheck();
    const un = listen<void>("tray:check-update", () => runCheck());
    return () => {
      un.then((fn) => fn());
    };
  }, [runCheck]);

  async function install() {
    if (!update) return;
    if (gpuBuild) {
      openUrl(`${RELEASES_URL}/tag/v${update.version}`).catch(console.error);
      return;
    }
    try {
      setStatus("downloading");
      setError(null);
      let downloaded = 0;
      let total = 0;
      await update.downloadAndInstall((event) => {
        if (event.event === "Started") {
          total = event.data.contentLength ?? 0;
          setProgress({ downloaded: 0, total });
        } else if (event.event === "Progress") {
          downloaded += event.data.chunkLength;
          setProgress({ downloaded, total });
        } else if (event.event === "Finished") {
          setStatus("installed");
        }
      });
      await relaunch();
    } catch (e) {
      console.error("updater install:", e);
      setError(String(e));
      setStatus("available");
    }
  }

  if (dismissed || status === "idle" || status === "checking") return null;

  const pct =
    progress && progress.total > 0
      ? Math.round((progress.downloaded / progress.total) * 100)
      : null;

  return (
    <div
      className={cn(
        "flex items-center gap-3 border-b bg-primary/5 px-4 py-2 text-sm",
      )}
    >
      <Download className="h-4 w-4 text-primary" />
      <div className="flex-1">
        {status === "available" && update && (
          <span>
            {t("updater.newVersion", { version: update.version })}
            {gpuBuild && (
              <span className="ml-2 text-xs text-muted-foreground">
                {t("updater.gpuBuildHint")}
              </span>
            )}
            {update.date && (
              <span className="ml-2 text-xs text-muted-foreground">
                ({update.date})
              </span>
            )}
          </span>
        )}
        {status === "downloading" && (
          <span>
            {t("updater.downloading", { percent: pct !== null ? ` (${pct}%)` : "" })}
          </span>
        )}
        {status === "installed" && <span>{t("updater.installed")}</span>}
        {error && (
          <p className="text-xs text-destructive">
            {t("common.error")}: {error}
          </p>
        )}
      </div>
      {status === "available" && (
        <Button size="sm" onClick={install}>
          <RefreshCw className="h-3.5 w-3.5" />
          {gpuBuild ? t("updater.openRelease") : t("updater.install")}
        </Button>
      )}
      <Button
        size="sm"
        variant="ghost"
        onClick={() => setDismissed(true)}
        title={t("updater.hide")}
      >
        <X className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
