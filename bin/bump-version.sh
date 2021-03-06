#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

if [ -n "$(git diff HEAD)" ]; then
    echo "Git repo must be clean"
    exit 4
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
version_file="$SCRIPTPATH/../bohr/VERSION"
bohr_config_file="$SCRIPTPATH/../test-b2b/scenario1/bohr.json"

pip install jq===1.1.2 -qqq

BOHR_VERSION="$( cat $version_file )"
echo "Current bohr version is $BOHR_VERSION"

if ! [[ "$BOHR_VERSION" =~ .*-rc ]]; then
    echo "Version in bohr/VERSION file should match regex: .*-rc"
    exit 3
fi

tag_bohr_version="$(echo $BOHR_VERSION | sed -e 's/-rc$//g' )"

jq ".bohr_framework_version = \"$tag_bohr_version\"" "$bohr_config_file" > "${bohr_config_file}.tmp"
mv -f "${bohr_config_file}.tmp" "$bohr_config_file"
echo "$tag_bohr_version" > "$version_file"

tag="v$tag_bohr_version"
echo "Creating tag: $tag"

git commit -am "Bump version to $tag"
git tag -a "$tag" -m "$tag"
git push --tags

patch_version="$(echo "$tag_bohr_version" | sed -E 's/.*([0-9]+)$/\1/g' )"
incremented_patch_version=$(echo "$patch_version+1" | bc )
echo $tag_bohr_version | sed -E "s/[0-9]+$/$incremented_patch_version-rc/g" > "$version_file"

new_bohr_version="$(cat $version_file)"
echo "New bohr version is $new_bohr_version"

jq ".bohr_framework_version = \"$new_bohr_version\"" "$bohr_config_file" > "${bohr_config_file}.tmp"
mv -f "${bohr_config_file}.tmp" "$bohr_config_file"

git commit -am "Bump version to $new_bohr_version"
git push
