#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# build_app.sh  —  Build FocusFeed.app (double-clickable macOS app bundle)
#
# Usage:
#   chmod +x build_app.sh
#   ./build_app.sh
#
# What it does:
#   1. Generates AppIcon.icns via Pillow + iconutil
#   2. Creates FocusFeed.app bundle structure
#   3. Writes a launcher shell script that activates conda and runs app.py
#   4. Sets correct permissions
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BOLD='\033[1m'; GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; R='\033[0m'
info()    { echo -e "${BOLD}▸  $*${R}"; }
success() { echo -e "${GREEN}✓  $*${R}"; }
warn()    { echo -e "${YELLOW}⚠  $*${R}"; }
die()     { echo -e "${RED}✕  $*${R}"; exit 1; }

APP_NAME="FocusFeed"
BUNDLE="${APP_NAME}.app"
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo ""
echo -e "${BOLD}Building ${APP_NAME}.app …${R}"
echo ""

# ── 0. Sanity checks ─────────────────────────────────────────────────────────
command -v conda    >/dev/null 2>&1 || die "conda not found. Install Miniconda first."
command -v iconutil >/dev/null 2>&1 || die "iconutil not found (macOS-only tool)."
conda env list | grep -q "^focusfeed " || die "conda env 'focusfeed' not found.\nRun:  conda env create -f environment.yml"

# ── 1. Generate icon ─────────────────────────────────────────────────────────
info "Generating icon…"
conda run -n focusfeed pip install Pillow -q 2>/dev/null || true
conda run -n focusfeed python make_icon.py
iconutil -c icns AppIcon.iconset -o AppIcon.icns
success "AppIcon.icns"

# ── 2. Clean & create bundle skeleton ────────────────────────────────────────
info "Creating bundle structure…"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS"
mkdir -p "$BUNDLE/Contents/Resources"

# ── 3. Copy icon ─────────────────────────────────────────────────────────────
cp AppIcon.icns "$BUNDLE/Contents/Resources/AppIcon.icns"

# ── 4. Info.plist ─────────────────────────────────────────────────────────────
cat > "$BUNDLE/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>FocusFeed</string>
  <key>CFBundleDisplayName</key>       <string>FocusFeed</string>
  <key>CFBundleIdentifier</key>        <string>com.focusfeed.app</string>
  <key>CFBundleVersion</key>           <string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleExecutable</key>        <string>FocusFeed</string>
  <key>CFBundleIconFile</key>          <string>AppIcon</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>NSHighResolutionCapable</key>   <true/>
  <key>LSMinimumSystemVersion</key>    <string>13.0</string>
</dict>
</plist>
PLIST

# ── 5. Launcher script ────────────────────────────────────────────────────────
LAUNCHER="$BUNDLE/Contents/MacOS/$APP_NAME"

cat > "$LAUNCHER" <<'LAUNCHER'
#!/usr/bin/env bash
# FocusFeed launcher
# Resolves project dir → activates conda → kills stale server → runs app.py

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# MacOS/ → Contents/ → FocusFeed.app/ → project dir
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_DIR" || {
    osascript -e 'display dialog "Could not locate the FocusFeed project folder.\n\nMake sure FocusFeed.app is inside the FocusFeed project directory." with title "FocusFeed" buttons {"OK"} with icon stop'
    exit 1
}

# ── Find conda ──────────────────────────────────────────────────
CONDA_SH=""
for candidate in \
    "$HOME/opt/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/opt/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
    "/opt/homebrew/Caskroom/anaconda/base/etc/profile.d/conda.sh" \
    "/usr/local/miniconda3/etc/profile.d/conda.sh" \
    "/usr/local/anaconda3/etc/profile.d/conda.sh"
do
    [ -f "$candidate" ] && CONDA_SH="$candidate" && break
done

if [ -z "$CONDA_SH" ]; then
    osascript -e 'display dialog "Conda was not found on this machine.\n\nInstall Miniconda from https://docs.conda.io/en/latest/miniconda.html\nthen run:  conda env create -f environment.yml" with title "FocusFeed — Setup Required" buttons {"OK"} with icon stop'
    exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH"

if ! conda activate focusfeed 2>/dev/null; then
    osascript -e 'display dialog "The \"focusfeed\" conda environment was not found.\n\nOpen Terminal in the FocusFeed folder and run:\n  conda env create -f environment.yml" with title "FocusFeed — Setup Required" buttons {"OK"} with icon stop'
    exit 1
fi

# ── Kill any stale server on port 8000 ─────────────────────────
lsof -ti :8000 | xargs kill -9 2>/dev/null || true

# ── Launch ──────────────────────────────────────────────────────
exec python app.py
LAUNCHER

chmod +x "$LAUNCHER"

# ── 6. Tell macOS about the new app ──────────────────────────────────────────
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f "$DIR/$BUNDLE" 2>/dev/null || true

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
success "Built  →  $DIR/$BUNDLE"
echo ""
echo -e "  ${BOLD}Double-click FocusFeed.app${R}  or run:  open FocusFeed.app"
echo ""
