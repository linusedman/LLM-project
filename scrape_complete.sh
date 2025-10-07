#!/usr/bin/env bash
# Simple scraper for Kontrollwiki detail pages:
# - Finds related EUR-Lex links and downloads CELEX/OJ PDFs
# - Also handles Livsmedelsverket links that end up on a PDF (globalassets or /link redirect)
set -u  # no '-e' or 'pipefail' so we don't abort the whole run

BASE="https://kontrollwiki.livsmedelsverket.se"
UA="Mozilla/5.0 (compatible; KontrollwikiScraper/1.5; +https://example.org)"
OUT="scraped_pdfs"
STEP=50
SLEEP=0.5

mkdir -p "$OUT"

# Fetch helper
fetch() { curl -sSL --fail --retry 3 --retry-delay 2 -A "$UA" "$@"; }

# Resolve final URL after redirects (used for Livsmedelsverket /link/... -> ...globalassets/...pdf)
final_url() { curl -sSL -A "$UA" -o /dev/null -w '%{url_effective}' "$1"; }

echo "Scanning pages and downloading PDFs (CELEX direct via /SV/TXT/PDF + Livsmedelsverket PDFs)…"
offset=0
while :; do
  PAGE_URL="$BASE/sok?typ=7&sidOffset=$offset"
  echo "• Page offset=$offset"

  html="$(fetch "$PAGE_URL" 2>/dev/null || true)"
  if [ -z "${html:-}" ]; then
    echo "  (stop: could not fetch page $PAGE_URL)"
    break
  fi
  sleep "$SLEEP"

  details_tmp="$(mktemp)"
  { printf "%s" "$html" \
      | htmlq 'div.row.sok-hits div.sok-resultat > a' --attribute href 2>/dev/null \
      | sed 's#^/##' \
      | sort -u \
      || true; } >"$details_tmp"

  if ! [ -s "$details_tmp" ]; then
    echo "  no more results. done."
    rm -f "$details_tmp"
    break
  fi

  # Iterate each detail page
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    detail_url="$BASE/$rel"
    echo "  - Detail: $detail_url"

    dhtml="$(fetch "$detail_url" 2>/dev/null || true)"
    if [ -z "${dhtml:-}" ]; then
      echo "    (skip detail: failed to fetch)"
      continue
    fi
    sleep "$SLEEP"

    # Collect candidate links from "Relaterad information"-block
    links_tmp="$(mktemp)"
    { printf "%s" "$dhtml" \
        | htmlq 'p.related-info a' --attribute href 2>/dev/null \
        | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' \
        | sort -u \
        || true; } >"$links_tmp"

    if ! [ -s "$links_tmp" ]; then
      echo "    • No related links found."
      rm -f "$links_tmp"
      continue
    fi

    # Process each link: EUR-Lex (CELEX/OJ) first, then Livsmedelsverket PDFs
    while IFS= read -r raw; do
      [ -n "$raw" ] || continue

      # Normalize basic percent-encoding for ":" so CELEX parsers work
      url="${raw//%3A/:}"

      if printf '%s' "$url" | grep -qi 'eur-lex\.europa\.eu'; then
        # --- EUR-Lex: try CELEX first
        celex="$(printf '%s\n' "$url" | sed -n 's/.*CELEX:\([0-9A-Z()\-]\{1,\}\).*/\1/p')"
        if [ -n "$celex" ]; then
          pdf_url="https://eur-lex.europa.eu/legal-content/SV/TXT/PDF/?uri=CELEX:${celex}"
          out="${OUT}/${celex}.pdf"
        else
          # --- EUR-Lex: OJ fallback (OJ:L_YYYYxxxxx etc.)
          oj="$(printf '%s\n' "$url" | sed -n 's/.*uri=OJ:\([A-Za-z0-9_.()-]\{1,\}\).*/\1/p')"
          if [ -n "$oj" ]; then
            pdf_url="https://eur-lex.europa.eu/legal-content/SV/TXT/PDF/?uri=OJ:${oj}"
            out="${OUT}/OJ_${oj}.pdf"
          else
            echo "    • No CELEX or OJ id in: $url"
            continue
          fi
        fi

        if [ -f "$out" ]; then
          echo "    ✓ Exists: $out"
        else
          if fetch "$pdf_url" -o "$out" 2>/dev/null; then
            echo "    ✓ Saved:  $out"
          else
            echo "    ✗ Failed: $pdf_url"
            rm -f "$out"
          fi
        fi
        sleep "$SLEEP"
        continue
      fi

      # --- Livsmedelsverket: direct PDF or /link/... that redirects to a PDF
      if printf '%s' "$url" | grep -qi 'livsmedelsverket\.se'; then
        # Follow redirects to find the real target (often /globalassets/...pdf)
        final="$(final_url "$url")"
        case "$final" in
          *.pdf|*.pdf\?*|*.PDF|*.PDF\?*)
            # Strip query/fragment and take the last path segment as filename
            fname="$(printf '%s' "$final" | sed 's/[?#].*$//' | awk -F/ '{print $NF}')"
            # Replace spaces with underscores, just in case
            fname="$(printf '%s' "$fname" | tr ' ' '_' )"
            out="${OUT}/${fname}"
            if [ -f "$out" ]; then
              echo "    ✓ Exists: $out"
            else
              if fetch "$final" -o "$out" 2>/dev/null; then
                echo "    ✓ Saved:  $out"
              else
                echo "    ✗ Failed: $final"
                rm -f "$out"
              fi
            fi
            ;;
          *)
            echo "    • Not a PDF after redirect: $final"
            ;;
        esac
        sleep "$SLEEP"
        continue
      fi

      # Other domains/links are ignored on purpose to keep this focused
      :
    done <"$links_tmp"
    rm -f "$links_tmp"

  done <"$details_tmp"
  rm -f "$details_tmp"

  offset=$((offset+STEP))
done

echo "All done."
