export type ReplayVerifyDevBypassInput = {
  flag?: string | null;
  hostname?: string | null;
  mode?: string | null;
  prod?: boolean;
};

function normalizedHostname(hostname: string | null | undefined): string {
  return (hostname ?? "").trim().toLowerCase().replace(/^\[(.*)\]$/, "$1");
}

export function isLoopbackHostname(hostname: string | null | undefined): boolean {
  const host = normalizedHostname(hostname);
  return host === "localhost" || host === "127.0.0.1" || host === "::1";
}

export function isReplayVerifyDevBypassAllowed(input: ReplayVerifyDevBypassInput): boolean {
  if ((input.flag ?? "").trim() !== "1") return false;
  if (!isLoopbackHostname(input.hostname)) return false;
  if (input.prod !== false) return false;
  if ((input.mode ?? "").trim().toLowerCase() === "production") return false;
  return true;
}

export function replayVerifyDevBypassFromRuntime(): boolean {
  if (typeof window === "undefined") return false;
  return isReplayVerifyDevBypassAllowed({
    flag: import.meta.env.VITE_REPLAY_VERIFY_DEV_BYPASS,
    hostname: window.location.hostname,
    mode: import.meta.env.MODE,
    prod: import.meta.env.PROD,
  });
}
