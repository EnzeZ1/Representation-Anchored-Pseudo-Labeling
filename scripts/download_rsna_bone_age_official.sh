#!/usr/bin/env bash
set -euo pipefail

umask 077
destination="${1:-/nobackup/enzez/data/rsna_bone_age/2017}"
mkdir -p "$destination"
chmod 0700 "$destination"

download() {
  local url="$1"
  local name="$2"
  touch "$destination/$name"
  chmod 0600 "$destination/$name"
  curl --fail --location --retry 10 --retry-delay 5 --continue-at - \
    --output "$destination/$name" "$url"
  chmod 0600 "$destination/$name"
}

download 'https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set.zip' \
  'Bone Age Training Set.zip'
download 'https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set+Annotations.zip' \
  'Bone Age Training Set Annotations.zip'
download 'https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Validation+Set.zip' \
  'Bone Age Validation Set.zip'
download 'https://www.rsna.org/artificial-intelligence/-/media/Files/RSNA/Education/AI%20resources%20and%20training/AI%20image%20challenge/RSNA-2017-Pediatric-Bone-Age-Challenge-Dataset-Description.ashx?hash=A0B423007088816AFFACDCA934E2F09F903215F4&la=en' \
  'RSNA-2017-Pediatric-Bone-Age-Challenge-Dataset-Description.pdf'
download 'https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/rsna-2017-ai-challenge-acknowledgements.pdf?hash=4CEAA028CBF623B10D7B83F9742D1ACD&rev=ecb8ef34b5564b79a7c695ff9c1e87b0' \
  'RSNA-2017-AI-Challenge-Acknowledgements.pdf'
download 'https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/rsna-2017-ai-challenge-terms-of-use-and-attribution_final.pdf?hash=59FCA62E83FE7C923DEABF61CB0F5A66&rev=5715628510254926bd674f01f78629d3' \
  'RSNA-2017-AI-Challenge-Terms-of-Use-and-Attribution.pdf'

sha256sum "$destination"/* > "$destination/SHA256SUMS.local"
chmod 0600 "$destination/SHA256SUMS.local"
