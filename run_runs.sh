job_names=(
    "channel_950m_x_350m_res1m_nlev100_vdiff00001"
    "channel_950m_x_350m_res1m_nlev100_vdiff00005"
    "channel_950m_x_350m_res1m_nlev100_vdiff00010"
    "channel_950m_x_350m_res1m_nlev100_vdiff00050"
    "channel_950m_x_350m_res1m_nlev100_vdiff00100"
    "channel_950m_x_350m_res1m_nlev100_vdiff00150"
    "channel_950m_x_350m_res1.25m_nlev80_vdiff00001"
    "channel_950m_x_350m_res1.25m_nlev80_vdiff00005"
    "channel_950m_x_350m_res1.25m_nlev80_vdiff00010"
    "channel_950m_x_350m_res1.25m_nlev80_vdiff00050"
    "channel_950m_x_350m_res1.25m_nlev80_vdiff00100"
    "channel_950m_x_350m_res1.25m_nlev80_vdiff00150"
    "channel_950m_x_350m_res1.5m_nlev64_vdiff00005"
    "channel_950m_x_350m_res1.5m_nlev64_vdiff0001"
    "channel_950m_x_350m_res1.5m_nlev64_vdiff0005"
    "channel_950m_x_350m_res1.5m_nlev64_vdiff0010"
)
for job_name in "${job_names[@]}"; do
    command="./run_icon.sh icon4py true false $job_name"
    #command="./run_icon.sh icon4py false true $job_name"
    echo "$command"
    eval "$command"
done
