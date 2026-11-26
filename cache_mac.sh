
#!/usr/bin/env bash

set -e

CACHE_DIR="/goinfre/mbouderr/.cache/uv"

echo "Setting up uv cache..."

# Create cache directory
mkdir -p "$CACHE_DIR"

# Configure uv cache permanently for Fish
if command -v fish >/dev/null 2>&1; then
    fish -c "set -Ux UV_CACHE_DIR '$CACHE_DIR'"
fi

# Configure for Bash
if [ -f "$HOME/.bashrc" ]; then
    grep -qxF "export UV_CACHE_DIR=\"$CACHE_DIR\"" "$HOME/.bashrc" ||
        echo "export UV_CACHE_DIR=\"$CACHE_DIR\"" >> "$HOME/.bashrc"
fi

echo "Cache directory: $CACHE_DIR"
echo "UV_CACHE_DIR configured successfully!"

ls -ld "$CACHE_DIR"
