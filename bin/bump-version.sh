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

if ! [[ "$BOHR_VERSION" =~ [0-9]+\\.[0-9]+\\.[0-9]+-rc ]]; then
    echo "Version in bohr/VERSION file should match regex: [0-9]+\.[0-9]+\.[0-9]+-rc"
    exit 3
fi

version_wo_rc="$(echo $BOHR_VERSION | sed -e 's/-rc$//g' )"
patch_version="$(echo "$version_wo_rc" | sed -E 's/^([0-9]+)\.([0-9]+)\.([0-9]+)$/\3/g' )"
minor_version="$(echo "$version_wo_rc" | sed -E 's/^([0-9]+)\.([0-9]+)\.([0-9]+)$/\2/g' )"
major_version="$(echo "$version_wo_rc" | sed -E 's/^([0-9]+)\.([0-9]+)\.([0-9]+)$/\1/g' )"

if [[ ${1:-patch} == "minor" ]]; then
    minor_version=$(echo "$minor_version+1" | bc)
    patch_version=0
elif [[ ${1:-patch} == "major" ]]; then
    major_version=$(echo "$major_version+1" | bc)
    minor_version=0
    patch_version=0
fi

tag_bohr_version="$major_version.$minor_version.$patch_version"

jq ".bohr_framework_version = \"$tag_bohr_version\"" "$bohr_config_file" > "${bohr_config_file}.tmp"
mv -f "${bohr_config_file}.tmp" "$bohr_config_file"
echo "$tag_bohr_version" > "$version_file"

tag="v$tag_bohr_version"
echo "Creating tag: $tag"

git commit -am "Bump version to $tag"
git tag -a "$tag" -m "$tag"
git push --tags

incremented_patch_version=$(echo "$patch_version+1" | bc )
echo $tag_bohr_version | sed -E "s/[0-9]+$/$incremented_patch_version-rc/g" > "$version_file"

new_bohr_version="$(cat $version_file)"
echo "New bohr version is $new_bohr_version"

jq ".bohr_framework_version = \"$new_bohr_version\"" "$bohr_config_file" > "${bohr_config_file}.tmp"
mv -f "${bohr_config_file}.tmp" "$bohr_config_file"

git commit -am "Bump version to $new_bohr_version"
git push
