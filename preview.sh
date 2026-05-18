#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v brew >/dev/null 2>&1; then
  brew_prefix="$(brew --prefix)"
  ruby_bin="$brew_prefix/opt/ruby/bin"
  imagemagick_bin="$brew_prefix/opt/imagemagick/bin"

  if [[ -d "$ruby_bin" ]]; then
    export PATH="$ruby_bin:$PATH"
  fi

  if [[ -d "$imagemagick_bin" ]]; then
    export PATH="$imagemagick_bin:$PATH"
  fi
fi

cd "$repo_root"

preview_dest="${PREVIEW_DEST:-${TMPDIR:-/tmp}/kamichanw-github-preview-site}"
mkdir -p "$preview_dest"
server_port="${JEKYLL_PORT:-4000}"
live_reload_port="${LIVERELOAD_PORT:-35729}"

while ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${server_port}$"; do
  server_port=$((server_port + 1))
done

while ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${live_reload_port}$"; do
  live_reload_port=$((live_reload_port + 1))
done

required_bundler_version="$(
  ruby -e '
    lockfile = "Gemfile.lock"
    if File.exist?(lockfile)
      content = File.read(lockfile)
      match = content.match(/BUNDLED WITH\s+([\d.]+)/m)
      puts(match[1]) if match
    end
  '
)"

bundle_cmd=(bundle)
if [[ -n "$required_bundler_version" ]]; then
  if ! bundle "_${required_bundler_version}_" --version >/dev/null 2>&1; then
    echo "Bundler ${required_bundler_version} is required by Gemfile.lock." >&2
    echo "Install it with: gem install bundler -v ${required_bundler_version}" >&2
    exit 1
  fi
  bundle_cmd=(bundle "_${required_bundler_version}_")
fi

"${bundle_cmd[@]}" config set --local path vendor/bundle >/dev/null

if ! "${bundle_cmd[@]}" check >/dev/null 2>&1; then
  "${bundle_cmd[@]}" install
fi

echo "Preview output: $preview_dest"
echo "Preview URL: http://127.0.0.1:${server_port}"
echo "LiveReload port: ${live_reload_port}"

exec "${bundle_cmd[@]}" exec jekyll serve \
  --config _config.yml,_config_local.yml \
  --destination "$preview_dest" \
  --host 127.0.0.1 \
  --port "$server_port" \
  --livereload \
  --livereload-port "$live_reload_port"
