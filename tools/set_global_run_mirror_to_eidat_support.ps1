param(
  [Parameter(Mandatory = $false)]
  [string]$ScannerEnvPath = ".\\user_inputs\\scanner.local.env",

  [Parameter(Mandatory = $false)]
  [string]$SupportDirName = "EIDAT Support",

  [Parameter(Mandatory = $false)]
  [string]$LinkPath = ".\\global_run_mirror",

  [Parameter(Mandatory = $false)]
  [switch]$CreateInsideIfLocked
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Read-EnvFileValue([string]$Path, [string]$Key) {
  if (-not (Test-Path -LiteralPath $Path)) { return $null }
  foreach ($raw in Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue) {
    if ($null -eq $raw) { $raw = "" }
    $line = ([string]$raw).Trim()
    if ($line.Length -eq 0) { continue }
    if ($line.StartsWith("#") -or $line.StartsWith(";")) { continue }
    $eq = $line.IndexOf("=")
    if ($eq -lt 1) { continue }
    $k = $line.Substring(0, $eq).Trim()
    if ($k -ne $Key) { continue }
    $v = $line.Substring($eq + 1).Trim()
    if ($v.Length -eq 0) { return $null }
    $hash = $v.IndexOf("#")
    if ($hash -ge 0) { $v = $v.Substring(0, $hash).Trim() }
    $semi = $v.IndexOf(";")
    if ($semi -ge 0) { $v = $v.Substring(0, $semi).Trim() }
    if ($v.Length -eq 0) { return $null }
    return $v
  }
  return $null
}

$repoRoot = Read-EnvFileValue -Path $ScannerEnvPath -Key "REPO_ROOT"
if (-not $repoRoot) {
  $fallback = ".\\user_inputs\\scanner.env"
  if ($ScannerEnvPath -ne $fallback) {
    $repoRoot = Read-EnvFileValue -Path $fallback -Key "REPO_ROOT"
    if ($repoRoot) { $ScannerEnvPath = $fallback }
  }
}
if (-not $repoRoot) { throw "REPO_ROOT not found in $ScannerEnvPath" }

$repoRootFull = [System.IO.Path]::GetFullPath($repoRoot)
$supportDir = Join-Path $repoRootFull $SupportDirName
if (-not (Test-Path -LiteralPath $supportDir)) {
  throw "Support dir not found: $supportDir"
}

$linkFull = [System.IO.Path]::GetFullPath($LinkPath)
$linkName = Split-Path -Leaf $linkFull
$linkParent = Split-Path -Parent $linkFull
if (-not (Test-Path -LiteralPath $linkParent)) {
  throw "Link parent folder not found: $linkParent"
}

if (Test-Path -LiteralPath $linkFull) {
  $item = Get-Item -LiteralPath $linkFull -Force
  $isReparse = ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
  if ($isReparse) {
    Remove-Item -LiteralPath $linkFull -Force
  } else {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $backup = Join-Path $linkParent ($linkName + "__backup_" + $ts)
    try {
      Rename-Item -LiteralPath $linkFull -NewName (Split-Path -Leaf $backup)
      Write-Host ("Backed up existing '" + $linkName + "' to: " + $backup)
    } catch {
      if (-not $CreateInsideIfLocked) {
        throw
      }
      # Fallback: keep existing folder and create a junction *inside* it.
      $inner = Join-Path $linkFull "_EIDAT_SUPPORT"
      if (-not (Test-Path -LiteralPath $inner)) {
        New-Item -ItemType Junction -Path $inner -Target $supportDir | Out-Null
      }
      $targetTxt = Join-Path $linkFull "_EIDAT_SUPPORT_TARGET.txt"
      $lines = @(
        "# EIDAT Support target (fallback: created inside existing global_run_mirror)",
        ("# Generated: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")),
        ("REPO_ROOT=" + $repoRootFull),
        ("EIDAT_SUPPORT=" + ([System.IO.Path]::GetFullPath($supportDir)))
      )
      [System.IO.File]::WriteAllText($targetTxt, ($lines -join "`n") + "`n", (New-Object System.Text.UTF8Encoding($false)))
      Write-Host ("Created junction inside existing folder: " + $inner)
      Write-Host ("Wrote: " + $targetTxt)
      return
    }
  }
}

New-Item -ItemType Junction -Path $linkFull -Target $supportDir | Out-Null

$targetTxt = Join-Path $linkParent ($linkName + "_TARGET.txt")
$lines = @(
  "# global_run_mirror target",
  ("# Generated: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")),
  ("REPO_ROOT=" + $repoRootFull),
  ("EIDAT_SUPPORT=" + ([System.IO.Path]::GetFullPath($supportDir)))
)
[System.IO.File]::WriteAllText($targetTxt, ($lines -join "`n") + "`n", (New-Object System.Text.UTF8Encoding($false)))

Write-Host ("Junction created: " + $linkFull)
Write-Host (" -> " + ([System.IO.Path]::GetFullPath($supportDir)))
Write-Host ("Wrote: " + $targetTxt)
