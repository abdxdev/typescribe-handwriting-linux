#!/usr/bin/env bash
# Installs (or removes) the TypeScribe launcher for the current user so it shows
# up in the GNOME app grid / any freedesktop-compliant launcher.
#
#   ./install-desktop.sh              install
#   ./install-desktop.sh --uninstall  remove
set -euo pipefail

APP_ID="io.github.typescribe.TypeScribe"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
DESKTOP_FILE="$DATA_HOME/applications/$APP_ID.desktop"
ICON_FILE="$DATA_HOME/icons/hicolor/512x512/apps/$APP_ID.png"

refresh_caches() {
    update-desktop-database -q "$DATA_HOME/applications" 2>/dev/null || true
    gtk-update-icon-cache -q -t "$DATA_HOME/icons/hicolor" 2>/dev/null || true
}

if [[ "${1:-}" == "--uninstall" ]]; then
    rm -f "$DESKTOP_FILE" "$ICON_FILE"
    refresh_caches
    echo "Removed TypeScribe launcher."
    exit 0
fi

mkdir -p "$(dirname "$DESKTOP_FILE")" "$(dirname "$ICON_FILE")"

# Square the icon if ImageMagick is available, otherwise copy it as-is.
if command -v magick >/dev/null 2>&1; then
    magick "$APP_DIR/assets/typescribe.png" -background none -gravity center -extent "%[fx:max(w,h)]x%[fx:max(w,h)]" "$ICON_FILE"
else
    cp "$APP_DIR/assets/typescribe.png" "$ICON_FILE"
fi

# The GUI needs the distro's Python (PyGObject); the model runs in the conda env
# and is started by the GUI itself.
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=TypeScribe
GenericName=Text to Handwriting
Comment=Turn typed text into realistic handwriting
Exec=/usr/bin/python3 $APP_DIR/gui.py
Path=$APP_DIR
Icon=$APP_ID
Terminal=false
Categories=Utility;GTK;
Keywords=handwriting;text;writing;notes;
StartupNotify=true
StartupWMClass=$APP_ID
EOF

refresh_caches
echo "Installed: $DESKTOP_FILE"
echo "Launch TypeScribe from your app launcher, or run: /usr/bin/python3 $APP_DIR/gui.py"
