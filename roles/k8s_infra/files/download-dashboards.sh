#!/bin/sh
set -e

apk add --no-cache curl sed jq

mkdir -p /var/lib/grafana/dashboards

# --- Persistent dashboard cache (two-step: raw cache + always-patch) -
# The grafana-dashboards volume is a hostPath that survives pod restarts.
# IMPORTANT: the TTL cache lives on the RAW upstream download (in
# $SRC_DIR), NOT on the transformed result. The sed/jq patch is
# re-applied on EVERY run, so editing a patch always takes effect on the
# next restart while a warm raw is not re-downloaded. (Previously the TTL
# guarded the post-jq file, so a patch change was silently ignored until
# the cache expired.) A failed/offline pull (cluster is partly airgapped)
# never clobbers a good cached raw or a good provisioned dashboard.
DASH_DIR=/var/lib/grafana/dashboards
SRC_DIR="$DASH_DIR/src"          # raw upstream JSON, pre-transform
mkdir -p "$SRC_DIR"
# Re-fetch a cached RAW download only if older than this many days.
#   N>0 : refresh after N days  |  0 : always re-fetch  |  -1 : cache forever
CACHE_TTL_DAYS="${GRAFANA_DASHBOARD_CACHE_TTL_DAYS:-7}"

# fetch_src <name> <url>: ensure $SRC_DIR/<name> holds the raw upstream
# JSON. TTL-cached; atomic; keep-on-failure. All chatter to stderr so it
# never pollutes the JSON on stdout. Returns 0 if a usable raw exists.
fetch_src() {
  _name="$1"; _url="$2"; _s="$SRC_DIR/$_name"
  if [ "$CACHE_TTL_DAYS" != "0" ] && [ -s "$_s" ] && jq empty "$_s" >/dev/null 2>&1 \
     && { [ "$CACHE_TTL_DAYS" = "-1" ] || [ -z "$(find "$_s" -mtime +"$CACHE_TTL_DAYS" 2>/dev/null)" ]; }; then
    return 0   # warm raw within TTL — skip the network pull
  fi
  _t="$SRC_DIR/.$_name.$$.tmp"
  if curl -fsS --connect-timeout 5 --max-time 60 "$_url" -o "$_t" \
     && [ -s "$_t" ] && jq empty "$_t" >/dev/null 2>&1; then
    mv -f "$_t" "$_s"; echo "  fetched raw $_name" >&2; return 0
  fi
  rm -f "$_t"
  [ -s "$_s" ] && { echo "  WARN: fetch of $_name failed/invalid — keeping cached raw" >&2; return 0; }
  echo "  ERROR: fetch of $_name failed and no cached raw exists" >&2; return 1
}

# raw <name> <url>: emit the (cached or freshly-fetched) raw upstream JSON
# to stdout for the transform pipeline. Used as:
#   raw X.json "<url>" | sed … | jq … | commit X.json
# On fetch failure with no cached raw it emits nothing → the downstream
# commit keeps the previous good provisioned copy.
raw() {
  fetch_src "$1" "$2" || return 1
  cat "$SRC_DIR/$1"
}

# commit <file>: read a freshly-built dashboard from stdin, validate it is
# non-empty valid JSON, then atomically replace the provisioned copy. On
# any failure the existing file is kept — a bad build never wins. Always
# returns 0 so `set -e` and the pipeline don't abort on a failure.
commit() {
  _f="$DASH_DIR/$1"; _t="$DASH_DIR/.$1.$$.tmp"
  cat > "$_t"
  if [ -s "$_t" ] && jq empty "$_t" >/dev/null 2>&1; then
    mv -f "$_t" "$_f"; echo "  updated $1"
  else
    rm -f "$_t"
    [ -s "$_f" ] && echo "  WARN: build of $1 failed/invalid — keeping previous copy" \
                 || echo "  ERROR: build of $1 failed and no previous copy exists"
  fi
  return 0
}
# --------------------------------------------------------------------

# Dashboard downloads with datasource variable replacements
# https://github.com/dotdc/grafana-dashboards-kubernetes?tab=readme-ov-file#install-manually

# CronJobs dashboard (14279)
raw cronjobs-14279.json "https://grafana.com/api/dashboards/14279/revisions/latest/download" \
  | sed 's/${DS_PROMETHEUS}/Prometheus/g' \
  | commit cronjobs-14279.json

# Kubernetes dashboard (15661)
raw kubernetes-15661.json "https://grafana.com/api/dashboards/15661/revisions/latest/download" \
  | sed 's/${DS__VICTORIAMETRICS-PROD-ALL}/Prometheus/g' \
  | commit kubernetes-15661.json

# https://github.com/dotdc/grafana-dashboards-kubernetes?tab=readme-ov-file#install-manually
# Additional dashboards from dotdc/grafana-dashboards-kubernetes
# Format: "ID:filename"
DOTDC_DASHBOARDS="
  19105:k8s-addons-prometheus
  16337:k8s-addons-trivy-operator
  15761:k8s-system-api-server
  15762:k8s-system-coredns
  15757:k8s-views-global
  15758:k8s-views-namespaces
  15759:k8s-views-nodes
  15760:k8s-views-pods
"

for entry in $DOTDC_DASHBOARDS; do
  id="${entry%%:*}"
  name="${entry#*:}"
  echo "Downloading dashboard ${name} (${id})..."
  if [ "$id" = "15759" ]; then
    # k8s-views-nodes dashboard fixes:
    # 1. Replace instance="$instance" with node="$node" for node-exporter metrics (CPU, RAM, etc.)
    #    Our node-exporter has node label with hostname, but dashboard queries use instance
    # 2. Replace node="$node" with instance="$node" for kubelet_volume_stats metrics (PV panels)
    #    Our kubelet metrics have instance label with hostname, but dashboard queries use node
    # JSON contains escaped quotes: =\"$var\"
    raw "${name}-${id}.json" "https://grafana.com/api/dashboards/${id}/revisions/latest/download" \
      | sed 's/${DS_PROMETHEUS}/Prometheus/g' \
      | sed 's/instance=\\"$instance\\"/node=\\"$node\\"/g' \
      | sed 's/kubelet_volume_stats_\([a-z_]*\){node=\\"$node\\"/kubelet_volume_stats_\1{instance=\\"$node\\"/g' \
      | sed 's/cluster=\\"$cluster\\", //g' \
      | sed 's/, cluster=\\"$cluster\\"//g' \
      | commit "${name}-${id}.json"
  else
    # Strip cluster="$cluster" filter — single-cluster setup has no cluster label.
    # Handle both mid-selector (with trailing comma) and end-of-selector (with leading comma).
    raw "${name}-${id}.json" "https://grafana.com/api/dashboards/${id}/revisions/latest/download" \
      | sed 's/${DS_PROMETHEUS}/Prometheus/g' \
      | sed 's/cluster=\\"$cluster\\", //g' \
      | sed 's/, cluster=\\"$cluster\\"//g' \
      | commit "${name}-${id}.json"
  fi
done

# NUT UPS dashboard for DRuggeri nut_exporter (19308)
# Patches:
# - Manufacturer/Model: use "labelsToFields" transformation so the stat panel
#   can display the mfr/model Prometheus label as a field value. Values come
#   from NUT overrides (override.device.mfr/model in ups.conf).
# - Beeper: remove misleading noValue="0" default
# - Power/Runtime/realpower.nominal all work natively via NUT config.
# - ups template variable: enable includeAll + multi for "All" dropdown option
# - Query selectors: ups="$ups" → ups=~"$ups" for multi-value/All regex matching
#   The ups label is injected by Prometheus via per-UPS scrape jobs.
raw nut-ups-19308.json "https://grafana.com/api/dashboards/19308/revisions/latest/download" \
  | sed 's/ups=\\"$ups\\"/ups=~\\"$ups\\"/g; s/ups=\\"${ups}\\"/ups=~\\"${ups}\\"/g' \
  | jq '
    .panels |= map(
      if .title == "Manufacturer" then
        .transformations = [{"id":"labelsToFields","options":{}}]
        | .options.reduceOptions = {"calcs":["lastNotNull"],"fields":"/^mfr$/","values":false}
        | .options.textMode = "value"
      elif .title == "Model" then
        .transformations = [{"id":"labelsToFields","options":{}}]
        | .options.reduceOptions = {"calcs":["lastNotNull"],"fields":"/^model$/","values":false}
        | .options.textMode = "value"
      elif .title == "Status" then
        .targets[0].legendFormat = "{{ups}}: {{flag}}"
      elif .title == "Beeper Status" then
        .fieldConfig.defaults |= del(.noValue)
      else . end
    )
    | .templating.list |= map(
        if .name == "ups" then .includeAll = true | .multi = true
        else . end
      )
    | .panels += [
        {
          "id": 100, "type": "row", "title": "Combined UPS Summary",
          "collapsed": false,
          "gridPos": {"h": 1, "w": 24, "x": 0, "y": 31},
          "panels": []
        },
        {
          "id": 101, "type": "stat", "title": "Total Power",
          "gridPos": {"h": 4, "w": 6, "x": 0, "y": 32},
          "datasource": {"type": "prometheus", "uid": "prometheus"},
          "targets": [{"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "sum(network_ups_tools_ups_realpower_nominal * network_ups_tools_ups_load / 100)", "instant": true, "refId": "A"}],
          "fieldConfig": {"defaults": {"unit": "watt", "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 1350}, {"color": "red", "value": 1620}]}, "min": 0, "max": 1800}, "overrides": []},
          "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false}, "colorMode": "value", "graphMode": "area", "textMode": "auto", "orientation": "auto", "justifyMode": "auto"}
        },
        {
          "id": 102, "type": "stat", "title": "Total Capacity",
          "gridPos": {"h": 4, "w": 6, "x": 6, "y": 32},
          "datasource": {"type": "prometheus", "uid": "prometheus"},
          "targets": [{"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "sum(network_ups_tools_ups_realpower_nominal)", "instant": true, "refId": "A"}],
          "fieldConfig": {"defaults": {"unit": "watt", "thresholds": {"mode": "absolute", "steps": [{"color": "blue", "value": null}]}}, "overrides": []},
          "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false}, "colorMode": "value", "graphMode": "none", "textMode": "auto", "orientation": "auto", "justifyMode": "auto"}
        },
        {
          "id": 103, "type": "stat", "title": "Min Runtime",
          "gridPos": {"h": 4, "w": 6, "x": 12, "y": 32},
          "datasource": {"type": "prometheus", "uid": "prometheus"},
          "targets": [{"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "min(network_ups_tools_battery_runtime)", "instant": true, "refId": "A"}],
          "fieldConfig": {"defaults": {"unit": "s", "thresholds": {"mode": "absolute", "steps": [{"color": "red", "value": null}, {"color": "orange", "value": 300}, {"color": "green", "value": 600}]}}, "overrides": []},
          "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false}, "colorMode": "value", "graphMode": "area", "textMode": "auto", "orientation": "auto", "justifyMode": "auto"}
        },
        {
          "id": 104, "type": "stat", "title": "Min Charge",
          "gridPos": {"h": 4, "w": 6, "x": 18, "y": 32},
          "datasource": {"type": "prometheus", "uid": "prometheus"},
          "targets": [{"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "min(network_ups_tools_battery_charge)", "instant": true, "refId": "A"}],
          "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100, "thresholds": {"mode": "absolute", "steps": [{"color": "red", "value": null}, {"color": "orange", "value": 15}, {"color": "green", "value": 30}]}}, "overrides": []},
          "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false}, "colorMode": "value", "graphMode": "area", "textMode": "auto", "orientation": "auto", "justifyMode": "auto"}
        }
      ]
  ' | commit nut-ups-19308.json

# NVIDIA DCGM Exporter dashboard (12239)
# Patches:
# - Datasource: ${DS_PROMETHEUS} → "Prometheus" (our provisioned name)
# - Templating "gpu": original query "label_values(gpu)" returns empty
#   in Grafana 12 — needs a metric reference. Use DCGM_FI_DEV_GPU_TEMP.
# - Drop __inputs/__requires/iteration/gnetId; id:null so provisioning owns it.
# - Drop "GPU Framebuffer Mem Used" and "Tensor Core Utilization" panels:
#   on GB10/Spark (Grace-Blackwell unified memory) DCGM_FI_DEV_FB_USED is
#   not exported (no discrete framebuffer) and DCGM_FI_PROF_* profiling
#   metrics aren't enabled by the DCGM module on Spark hardware.
# - Migrate panels from Grafana 6.x schema:
#   * "graph" panel type (removed in Grafana 11) → "timeseries"; lift
#     yaxes[0].format/min/max → fieldConfig.defaults.{unit,min,max}.
#   * "gauge" panels use deprecated options.fieldOptions.defaults →
#     promote to fieldConfig.defaults + options.reduceOptions.
# - Per-pod-attribution dedup: dcgm-exporter's K8s integration USED to emit
#   one series per GPU-holding pod (exporter itself + every time-sliced
#   consumer like sglang/ollama) — identical gpu+instance, different
#   pod/container labels — so the panels showed duplicate legend rows and
#   sum() double-counted. Fixed at TWO layers (2026-06-05):
#     (a) SOURCE: dcgm_exporter.yml now runs DCGM_EXPORTER_KUBERNETES=false
#         → exactly one series per physical GPU, stable across pod restarts
#         (the dcgm scrape adds no `pod` label, so identity is restart-safe).
#     (b) DASHBOARD (here, belt-and-suspenders): wrap every DCGM selector in
#         `max by (gpu, instance)(…)`. Collapses any residual duplicate
#         series (e.g. if the K8s integration is ever re-enabled, or two
#         exporter pods overlap during a restart's staleness window) into
#         one line per GPU, and keeps avg()/sum() gauges from double-counting.
#   The old `,container="dcgm-exporter"` selector injection is REMOVED — with
#   K8s integration off there is no `container` label, so that filter would
#   match nothing and blank every panel.
raw dcgm-exporter-12239.json "https://grafana.com/api/dashboards/12239/revisions/latest/download" \
  | sed 's/${DS_PROMETHEUS}/Prometheus/g' \
  | jq '
    def migrate_graph:
      . + {
        type: "timeseries",
        fieldConfig: {
          defaults: {
            unit: (.yaxes[0].format // "short"),
            min: (.yaxes[0].min // null),
            max: (.yaxes[0].max // null),
            custom: {
              drawStyle: "line",
              lineWidth: (.linewidth // 1),
              fillOpacity: (if (.fill // 0) > 0 then 10 else 0 end),
              showPoints: "never",
              spanNulls: true
            }
          },
          overrides: []
        },
        options: {
          tooltip: { mode: (if .tooltip.shared then "multi" else "single" end) },
          legend: {
            displayMode: (if .legend.alignAsTable then "table" else "list" end),
            placement: (if .legend.rightSide then "right" else "bottom" end),
            showLegend: (.legend.show // true),
            calcs: ([
              (if .legend.avg then "mean" else empty end),
              (if .legend.current then "lastNotNull" else empty end),
              (if .legend.max then "max" else empty end),
              (if .legend.min then "min" else empty end)
            ])
          }
        }
      }
      | del(.yaxes, .xaxis, .lines, .fill, .linewidth, .legend, .tooltip,
            .nullPointMode, .percentage, .points, .pointradius, .renderer,
            .seriesOverrides, .spaceLength, .stack, .steppedLine, .thresholds,
            .timeFrom, .timeRegions, .timeShift, .aliasColors, .bars,
            .dashLength, .dashes);
    def migrate_gauge:
      # Force calcs=lastNotNull on all gauges. Original dashboard uses
      # "mean" / "sum" which reduce over the dashboard time-window —
      # nonsensical for a "current value" gauge whose Prometheus query
      # already aggregates across series (sum(...) / avg(...)).
      # The original "GPU Power Total" sum-over-time showed ~567 W for
      # spark1 idle (real current sum is ~12 W) — pure rendering bug.
      . + {
        fieldConfig: {
          defaults: (.options.fieldOptions.defaults // {}),
          overrides: (.options.fieldOptions.overrides // [])
        },
        options: {
          reduceOptions: {
            calcs: ["lastNotNull"],
            fields: "",
            values: (.options.fieldOptions.values // false)
          },
          orientation: (.options.orientation // "auto"),
          showThresholdLabels: (.options.showThresholdLabels // false),
          showThresholdMarkers: (.options.showThresholdMarkers // true)
        }
      };
    del(.__inputs, .__requires, .iteration, .gnetId)
    | .id = null
    | .uid = "dcgm-exporter-12239"
    | .panels |= map(
        select(.title != "GPU Framebuffer Mem Used"
               and .title != "Tensor Core Utilization")
      )
    | .panels |= map(
        if .type == "graph" then migrate_graph
        elif .type == "gauge" then migrate_gauge
        else . end
      )
    | .templating.list |= map(
        if .name == "gpu" then
          .query = "label_values(DCGM_FI_DEV_GPU_TEMP, gpu)"
          | .definition = "label_values(DCGM_FI_DEV_GPU_TEMP, gpu)"
        else . end
      )
    | .panels |= map(
        .targets |= map(
          .expr |= gsub("(?<m>DCGM_FI_[A-Z0-9_]+\\{.*\\})";
                        "max by (gpu, instance) (" + .m + ")")
        )
      )
    | .panels |= map(
        .targets |= map(
          if .legendFormat == "GPU {{gpu}}" then
            .legendFormat = "GPU {{gpu}} ({{instance}})"
          else . end
        )
      )
  ' | commit dcgm-exporter-12239.json

# SGLang inference dashboard (official, from the SGLang repo:
# examples/monitoring/grafana/dashboards/json/sglang-dashboard.json).
# Panels: E2E latency (+heatmap), TTFT (+heatmap), running reqs, gen
# throughput, cache hit rate, queued reqs. Pinned to v0.5.12 (matches
# default_sglang_image). Patches:
# - Datasource: upstream hardcodes uid "ddyfngn31dg5cf" → our "prometheus".
# - id:null so file-provisioning owns it (stable uid "sglang-dashboard").
# - Every panel expr gets an {instance=~"$instance",model_name=~"$model_name"}
#   selector so the template vars actually FILTER (model_name is auto-
#   stamped by SGLang = served model → models served on one instance over
#   time stay separable). First gsub widens the heatmaps' existing
#   model_name selector to add instance; second adds the full selector to
#   bare metrics (handles the [rate-interval] and end-of-expr cases).
# - METRIC-NAME NORMALISATION: upstream's example JSON uses underscore
#   names (sglang_e2e_request_latency_…), but the SGLang server emits a
#   COLON namespace (sglang:e2e_request_latency_…) — querying underscore
#   matches nothing → every panel "No data". Third gsub rewrites each
#   already-selectored metric to {__name__=~"sglang[:_]<name>", <sel>}
#   so BOTH separators match: covers the colon-emitting default instance
#   AND any future/other instance that emits underscore. label_values()
#   below uses the same sglang[:_] regex for the same reason.
# - instance/model_name template vars: scope label_values() to a real
#   metric (bare label_values() returns empty in Grafana 12) + includeAll
#   + multi so the "All" option works.
# - INSTANCE NAME not pod IP: the "instance" var + every selector use the
#   `sglang_instance` label (the sglang_instances key, e.g. "default")
#   instead of the Prometheus `instance` label (pod IP:port). The label is
#   stamped on the head pod template (sglang_instance.yml) and flows in via
#   the kubernetes-pods labelmap. The 4 gauge panels also get
#   legendFormat={{sglang_instance}} so their legend shows
#   the name. REQUIRES the head pod to carry the label (redeploy sglang, or
#   `kubectl label pod <head> sglang_instance=<key>` for an existing pod).
# - UNITS: latency panels (E2E/TTFT + heatmaps) → "s" (metric is _seconds);
#   running/queued/throughput → "short"; cache hit rate → "percentunit"
#   (SGLang emits a 0–1 fraction; observed peak 0.6 = 60%).
# - Rename "End-to-End Request Latency" (+ Heatmap) → "Total Request Duration":
#   sglang:e2e_request_latency_seconds is the FULL request duration (queue +
#   prefill + complete decode), dominated by output length — a response time,
#   not a latency. TTFT panels keep their name (that IS the latency). Rename
#   runs LAST so the unit/legend stages above still match the original titles.
raw sglang-dashboard.json "https://raw.githubusercontent.com/sgl-project/sglang/v0.5.12/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json" \
  | sed 's/ddyfngn31dg5cf/prometheus/g' \
  | jq '
    .panels |= map(
      if .targets then
        .targets |= map(
          .expr |= (
            gsub("\\{model_name=~\"\\$model_name\"\\}";
                 "{sglang_instance=~\"$instance\", model_name=~\"$model_name\"}")
            | gsub("(?<m>sglang_[a-z0-9_]+)(?<a>[^={a-z0-9_]|$)";
                   .m + "{sglang_instance=~\"$instance\", model_name=~\"$model_name\"}" + .a)
            | gsub("sglang_(?<name>[a-z0-9_]+)\\{(?<sel>[^}]*)\\}";
                   "{__name__=~\"sglang[:_]" + .name + "\", " + .sel + "}")
          )
        )
      else . end
    )
    | .id = null
    | .templating.list |= map(
        if .name == "instance" then
          .query = {"qryType":1,"query":"label_values({__name__=~\"sglang[:_]num_running_reqs\"}, sglang_instance)","refId":"PrometheusVariableQueryEditor-VariableQuery"}
          | .definition = "label_values({__name__=~\"sglang[:_]num_running_reqs\"}, sglang_instance)"
          | .includeAll = true | .multi = true | .current = {"text":"All","value":"$__all"}
        elif .name == "model_name" then
          .query = {"qryType":1,"query":"label_values({__name__=~\"sglang[:_]num_running_reqs\"}, model_name)","refId":"PrometheusVariableQueryEditor-VariableQuery"}
          | .definition = "label_values({__name__=~\"sglang[:_]num_running_reqs\"}, model_name)"
          | .includeAll = true | .multi = true | .current = {"text":"All","value":"$__all"}
        else . end
      )
    | .panels |= map(
        if (.title == "End-to-End Request Latency" or .title == "End-to-End Request Latency(s) Heatmap"
            or .title == "Time-To-First-Token Latency" or .title == "Time-To-First-Token Seconds Heatmap")
          then .fieldConfig.defaults.unit = "s"
        elif .title == "Cache Hit Rate" then (.fieldConfig.defaults.unit = "percentunit" | .fieldConfig.defaults.max = 1 | .fieldConfig.defaults.min = 0)
        elif (.title == "Num Running Requests" or .title == "Number Queued Requests"
              or .title == "Token Generation Throughput (Tokens / S)")
          then .fieldConfig.defaults.unit = "short"
        else . end
      )
    | .panels |= map(
        if (.title == "Num Running Requests" or .title == "Number Queued Requests"
            or .title == "Cache Hit Rate" or .title == "Token Generation Throughput (Tokens / S)")
          then .targets |= map(.legendFormat = "{{sglang_instance}}")
        else . end
      )
    | (.. | objects | select(.title == "End-to-End Request Latency") | .title) |= "Total Request Duration"
    | (.. | objects | select(.title == "End-to-End Request Latency(s) Heatmap") | .title) |= "Total Request Duration Heatmap"
  ' | commit sglang-dashboard.json

# LiteLLM proxy dashboard (official, grafana.com 24965).
# Panels: request volume, spend, tokens, deployment success/failure +
# latency, TTFT. Template vars: job / model (instance dropped, see below). Patches:
# - Datasource: the dashboard parameterises via a "datasource" template
#   var referenced as ${datasource} (uid form) + the ${DS_PROMETHEUS}
#   input + a stray uid "ad8llmv" → all rewritten to our uid "prometheus",
#   then the now-unused "datasource" var is dropped (no empty picker).
# - Drop __inputs/__requires; id:null + stable uid so provisioning owns it.
# - Drop the "instance" template var + strip instance=~"$instance" from every
#   expr → panels aggregate over ALL litellm pods. instance is just the pod
#   IP:port (a replica discriminator); LiteLLM runs a single replica, so the
#   picker only ever showed one real pod (+ a stale one during rollover) and
#   added no value. Re-add if LiteLLM is ever scaled to multiple replicas.
# - Rename "Models Latency" → "Total Request Duration": the panel uses
#   litellm_llm_api_latency_metric = the FULL upstream call duration (queue +
#   prefill + complete decode), dominated by output length — a response time,
#   not a latency. The actual latency is the TTFT panel
#   (litellm_llm_api_time_to_first_token_metric), left as-is.
# - Spend tied to the time range: the "Total Spend per Team/User/Model/User
#   Agent" panels summed the bare litellm_spend_metric_total COUNTER → the
#   all-time cumulative spend, ignoring the dashboard time range. Wrap the metric
#   in increase(…[$__range]) so they show spend WITHIN the selected window (and
#   it's counter-reset-safe across pod restarts). "Spend Rate" is left untouched
#   (already a rate()); only titles starting "Total Spend per" are rewritten.
# NOTE: no per-panel dedup needed — the hermes-default alias is a router
# model_group_alias (see litellm_router_settings), NOT a duplicate
# model_list deployment, so litellm_deployment_state has one series per
# real backend (the "LLM Deployment Analytics" panel shows one tile).
raw litellm-24965.json "https://grafana.com/api/dashboards/24965/revisions/latest/download" \
  | sed 's/${datasource}/prometheus/g; s/${DS_PROMETHEUS}/prometheus/g; s/ad8llmv/prometheus/g' \
  | jq '
    del(.__inputs, .__requires)
    | .id = null
    | .uid = "litellm-24965"
    | .templating.list |= map(select(.name != "datasource" and .name != "instance"))
    | walk(if type == "string" then
             (gsub(", instance=~\"\\$instance\""; "")
              | gsub("instance=~\"\\$instance\", "; "")
              | gsub("\\{instance=~\"\\$instance\"\\}"; "{}"))
           else . end)
    | (.. | objects | select(.title == "Models Latency") | .title) |= "Total Request Duration"
    | (.. | objects | select((.title? // "") | startswith("Total Spend per")) | .targets[]?.expr) |=
        gsub("(?<m>litellm_spend_metric_total\\{[^}]*\\})"; "increase(" + .m + "[$__range])")
  ' | commit litellm-24965.json

echo "Dashboards downloaded successfully."