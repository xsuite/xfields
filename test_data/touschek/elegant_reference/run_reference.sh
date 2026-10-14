#!/usr/bin/env bash
set -euo pipefail

case_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
elegant_bin=${ELEGANT_BIN:-elegant}
sdds_bin_dir=${SDDS_BIN_DIR:-}
: "${RPN_DEFNS:?Set RPN_DEFNS to the SDDS defns.rpn file}"

if [[ -n "${sdds_bin_dir}" ]]; then
    export PATH="${sdds_bin_dir}:${PATH}"
fi

cd "${case_dir}"
"${elegant_bin}" toy_ring.ele > elegant.stdout.txt 2> elegant.stderr.txt

for output in toy_ring.twi toy_ring.scatter.sdds toy_ring.distribution.sdds; do
    if [[ ! -s "${output}" ]]; then
        echo "ELEGANT did not produce expected output: ${output}" >&2
        exit 1
    fi
done

sddsquery toy_ring.twi > toy_ring.twi.layout.txt 2>&1
sddsquery toy_ring.scatter.sdds > toy_ring.scatter.layout.txt 2>&1
sddsquery toy_ring.distribution.sdds > toy_ring.distribution.layout.txt 2>&1

sddsprintout toy_ring.twi -columns=ElementName -columns=s \
    -columns=betax -columns=alphax -columns=etax -columns=etaxp \
    -columns=betay -columns=alphay -columns=etay -columns=etayp \
    -noTitle -noLabels > twiss_table.txt
sddsprintout toy_ring.distribution.sdds \
    -parameters=Element_Name -parameters=s -parameters=Piwinski_AveR \
    -parameters=Piwinski_Rate -parameters=MC_Rate -parameters=Ignored_Rate \
    -noTitle > touschek_rates.txt
sddsprintout toy_ring.scatter.sdds \
    -parameters=Particles -parameters=pCentral -parameters=AveRate \
    -parameters=NScatter -parameters=PLocalRate -parameters=SLocalRate \
    -parameters=IgnoredRate -noTitle > scatter_summary.txt

{
    echo "element_name,s_m,piwinski_local_rate_hz,piwinski_average_rate_hz,mc_local_rate_hz,ignored_rate_hz,section_rate_hz,selected_count"
    paste -d, \
        <(sdds2stream toy_ring.distribution.sdds -parameters=Element_Name) \
        <(sdds2stream toy_ring.distribution.sdds -parameters=s) \
        <(sdds2stream toy_ring.distribution.sdds -parameters=Piwinski_Rate) \
        <(sdds2stream toy_ring.distribution.sdds -parameters=Piwinski_AveR) \
        <(sdds2stream toy_ring.distribution.sdds -parameters=MC_Rate) \
        <(sdds2stream toy_ring.distribution.sdds -parameters=Ignored_Rate) \
        <(sdds2stream toy_ring.scatter.sdds -parameters=NScatter) \
        <(sdds2stream toy_ring.scatter.sdds -parameters=Particles)
} > reference.csv

awk -F, '
    NR > 1 {total += $7; selected += $8}
    END {
        printf "{\n"
        printf "  \"bunch_intensity\": 4000000000,\n"
        printf "  \"total_piwinski_rate_hz\": %.15g,\n", total
        printf "  \"touschek_lifetime_s\": %.15g,\n", 4000000000 / total
        printf "  \"selected_scattered_particles\": %d\n", selected
        printf "}\n"
    }
' reference.csv > reference_summary.json

python_bin=${PYTHON_BIN:-python}
"${python_bin}" extract_scattered_distribution.py \
    --sdds2stream "$(command -v sdds2stream)"

sha256sum toy_ring.ele toy_ring.lte momentum_aperture.sdds \
    toy_ring.twi toy_ring.scatter.sdds toy_ring.distribution.sdds \
    twiss_table.txt touschek_rates.txt scatter_summary.txt \
    reference.csv reference_summary.json \
    scattered_distribution_reference.npz \
    scattered_distribution_metadata.json > SHA256SUMS
