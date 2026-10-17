// src/routes/index.tsx
import { createFileRoute } from "@tanstack/react-router";
import React, { useState, useEffect, useMemo } from "react";
import {
  Recycle,
  Sparkles,
  SlidersHorizontal,
  FileBraces,
  Upload,
  Scale,
  Box,
  Hash,
  Camera,
  Check,
  Leaf,
  Trash2,
  ArrowRight,
  Info,
  Factory,
  ArrowUpRight,
  Loader2,
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from "recharts";

export const Route = createFileRoute("/")({
  component: PolyRouteApp,
});

const API_BASE = "http://127.0.0.1:8501";

interface LaneSegment {
  route: string;
  label: string;
  kg: number;
  pct: number;
}

interface Lane {
  code: number;
  polymer: string;
  name: string;
  kg: number;
  low: number;
  high: number;
  width: number;
  segments: LaneSegment[];
  items?: number;
}

interface AnalysisResult {
  source: string;
  n_items: number;
  lanes: Lane[];
  saved: number;
  result: {
    total_kg?: number;
    plastic_kg?: number;
    kg: Record<string, number>;
    kg_ci?: Record<string, [number, number]>;
    item_counts?: Record<string, number>;
  };
  carbon: { label: string; value: number; width: number }[];
  route_labels: Record<string, string>;
  routes_used: string[];
  explanation?: string;
  notes?: string[];
  placeholders?: string[];
}

const ROUTE_COLORS: Record<string, string> = {
  recycle: "#059669",
  compost: "#65a30d",
  combust: "#d97706",
  landfill: "#57534e",
};

const POLYMER_COLORS = [
  "#0284c7",
  "#0d9488",
  "#7c3aed",
  "#db2777",
  "#ea580c",
  "#dc2626",
  "#64748b",
];

const FALLBACK_DATA: AnalysisResult = {
  source: "demo-mix (Illustrative)",
  n_items: 100,
  saved: 12.2,
  result: {
    total_kg: 50,
    plastic_kg: 31.0,
    kg: { PET: 13.0, HDPE: 8.4, PP: 5.9, Other: 3.7 },
  },
  route_labels: {
    recycle: "Mechanical recycling",
    compost: "Composting",
    combust: "Energy recovery",
    landfill: "Landfill",
  },
  routes_used: ["recycle", "compost", "combust", "landfill"],
  carbon: [
    { label: "Everything to landfill", value: 38.5, width: 50 },
    { label: "PolyRoute plan", value: 26.3, width: 34 },
  ],
  lanes: [
    {
      code: 1,
      polymer: "PET",
      name: "Polyethylene terephthalate",
      kg: 13.0,
      low: 11.5,
      high: 14.5,
      width: 100,
      segments: [
        { route: "recycle", label: "Mechanical recycling", kg: 9.2, pct: 71 },
        { route: "combust", label: "Energy recovery", kg: 3.8, pct: 29 },
      ],
    },
    {
      code: 2,
      polymer: "HDPE",
      name: "High-density polyethylene",
      kg: 8.4,
      low: 7.2,
      high: 9.6,
      width: 65,
      segments: [
        { route: "recycle", label: "Mechanical recycling", kg: 4.4, pct: 52 },
        { route: "compost", label: "Composting", kg: 4.0, pct: 48 },
      ],
    },
    {
      code: 5,
      polymer: "PP",
      name: "Polypropylene",
      kg: 5.9,
      low: 4.8,
      high: 7.0,
      width: 45,
      segments: [
        { route: "compost", label: "Composting", kg: 2.2, pct: 37 },
        { route: "combust", label: "Energy recovery", kg: 3.6, pct: 63 },
      ],
    },
    {
      code: 7,
      polymer: "Other",
      name: "Multilayer, PLA and others",
      kg: 3.7,
      low: 2.9,
      high: 4.5,
      width: 28,
      segments: [{ route: "landfill", label: "Landfill", kg: 3.7, pct: 100 }],
    },
  ],
};

function PolyRouteApp() {
  const [samples, setSamples] = useState<string[]>([
    "medium_100.json",
    "large_500.json",
  ]);
  const [sampleFile, setSampleFile] = useState("medium_100.json");
  const [classifierFile, setClassifierFile] = useState<File | null>(null);
  const [weightMode, setWeightMode] = useState<
    "measured" | "volume" | "count" | "photos"
  >("measured");

  const [totalKgInput, setTotalKgInput] = useState("50");
  const [volumeM3Input, setVolumeM3Input] = useState("1.5");
  const [itemCountInput, setItemCountInput] = useState("100");

  const [recycleCap, setRecycleCap] = useState("30");
  const [compostCap, setCompostCap] = useState("12");
  const [landfillCap, setLandfillCap] = useState("20");

  const [data, setData] = useState<AnalysisResult>(FALLBACK_DATA);
  const [isLive, setIsLive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeRouteIndex, setActiveRouteIndex] = useState<number | null>(null);
  const [activePolymerIndex, setActivePolymerIndex] = useState<number | null>(
    null
  );

  useEffect(() => {
    // 1. Set clean PolyRoute title
    document.title = "PolyRoute";

    // 2. Replace Lovable favicon with a clean green Recycle SVG icon
    const svgIcon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M7 19H4.815a1.83 1.83 0 0 1-1.57-.881 1.785 1.785 0 0 1-.004-1.784L7.196 9.5"/><path d="M11 19h8.203a1.83 1.83 0 0 0 1.556-.89 1.784 1.784 0 0 0 0-1.775l-1.226-2.12"/><path d="m14 16-3 3 3 3"/><path d="M8.293 13.596 7.196 9.5 3.1 10.598"/><path d="m9.344 5.811 1.093-1.892A1.83 1.83 0 0 1 11.985 3a1.784 1.784 0 0 1 1.546.888l3.943 6.843"/><path d="m13.378 9.633 4.096 1.098 1.097-4.096"/></svg>`;
    const iconUrl = `data:image/svg+xml,${encodeURIComponent(svgIcon)}`;

    document.querySelectorAll("link[rel*='icon']").forEach((el) => el.remove());
    const link = document.createElement("link");
    link.rel = "icon";
    link.type = "image/svg+xml";
    link.href = iconUrl;
    document.head.appendChild(link);

    // Remove Lovable author/twitter meta tags if present
    document
      .querySelectorAll('meta[name="author"], meta[name="twitter:site"]')
      .forEach((el) => el.remove());

    fetch(`${API_BASE}/api/samples`)
      .then((res) => res.json())
      .then((info) => {
        if (info.samples && info.samples.length > 0) {
          setSamples(info.samples);
          setSampleFile(info.samples[0]);
        }
        if (info.defaults) {
          setRecycleCap(String(info.defaults.recycle_cap_kg ?? 30));
          setCompostCap(String(info.defaults.compost_cap_kg ?? 12));
          setLandfillCap(String(info.defaults.landfill_cap_kg ?? 20));
        }
      })
      .catch(() => {});

    runAnalysis();
  }, []);

  const runAnalysis = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (weightMode === "photos") return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    if (classifierFile) {
      formData.append("classifier_file", classifierFile);
    } else {
      formData.append("sample_file", sampleFile);
    }

    formData.append("weight_mode", weightMode);
    if (weightMode === "measured") formData.append("total_kg", totalKgInput);
    if (weightMode === "volume") formData.append("volume_m3", volumeM3Input);
    if (weightMode === "count") formData.append("item_count", itemCountInput);

    formData.append("recycle_cap_kg", recycleCap);
    formData.append("compost_cap_kg", compostCap);
    formData.append("landfill_cap_kg", landfillCap);

    try {
      const response = await fetch(`${API_BASE}/api/analyse`, {
        method: "POST",
        body: formData,
      });
      const json = await response.json();
      if (!response.ok) {
        throw new Error(json.error || "Failed to analyze batch.");
      }
      setData(json);
      setIsLive(true);
    } catch (err: any) {
      setError(
        err.message ||
          "Backend offline at 127.0.0.1:8501 — showing illustrative preview."
      );
      setIsLive(false);
    } finally {
      setLoading(false);
    }
  };

  const {
    routePieData,
    polymerPieData,
    totalPlasticKg,
    startingBatchKg,
    conicShareStyles,
  } = useMemo(() => {
    const totals: Record<string, number> = {
      recycle: 0,
      compost: 0,
      combust: 0,
      landfill: 0,
    };

    data.lanes.forEach((lane) => {
      lane.segments.forEach((seg) => {
        totals[seg.route] = (totals[seg.route] || 0) + seg.kg;
      });
    });

    const sumPlastic =
      Object.values(totals).reduce((a, b) => a + b, 0) ||
      data.result?.plastic_kg ||
      1;

    const routeOrder = [
      {
        key: "recycle",
        defaultLabel: "Mechanical recycling",
        dotClass: "route-dot-recycle",
      },
      {
        key: "compost",
        defaultLabel: "Composting",
        dotClass: "route-dot-compost",
      },
      {
        key: "combust",
        defaultLabel: "Energy recovery",
        dotClass: "route-dot-recovery",
      },
      {
        key: "landfill",
        defaultLabel: "Landfill",
        dotClass: "route-dot-landfill",
      },
    ];

    const rPie = routeOrder.map((r) => {
      const kg = Number((totals[r.key] || 0).toFixed(1));
      const pct = Math.round((kg / sumPlastic) * 100);
      return {
        key: r.key,
        name: data.route_labels?.[r.key] || r.defaultLabel,
        value: kg,
        pct,
        color: ROUTE_COLORS[r.key] || "#64748b",
        dotClass: r.dotClass,
      };
    });

    const pPie = data.lanes.map((lane, idx) => {
      const pct = Math.round((lane.kg / sumPlastic) * 100);
      return {
        polymer: lane.polymer,
        name: lane.name,
        value: Number(lane.kg.toFixed(1)),
        pct,
        low: lane.low,
        high: lane.high,
        segments: lane.segments,
        color: POLYMER_COLORS[idx % POLYMER_COLORS.length],
        idx,
      };
    });

    const batchKg =
      data.result?.total_kg ||
      (weightMode === "measured" ? Number(totalKgInput) : sumPlastic) ||
      50;

    return {
      routePieData: rPie,
      polymerPieData: pPie,
      totalPlasticKg: sumPlastic,
      startingBatchKg: batchKg,
      conicShareStyles: {
        "--share-recycle": `${rPie[0]?.pct ?? 44}%`,
        "--share-compost": `${rPie[1]?.pct ?? 20}%`,
        "--share-recovery": `${rPie[2]?.pct ?? 24}%`,
      } as React.CSSProperties,
    };
  }, [data, totalKgInput, weightMode]);

  return (
    <div className="app-shell no-scroll-viewport">
      <style>{`
        html, body {
          height: 100vh !important;
          max-height: 100vh !important;
          overflow: hidden !important;
          margin: 0 !important;
          padding: 0 !important;
        }
        .no-scroll-viewport {
          height: 100vh !important;
          max-height: 100vh !important;
          display: flex !important;
          flex-direction: column !important;
          overflow: hidden !important;
          box-sizing: border-box !important;
        }
        .no-scroll-viewport .app-header {
          flex-shrink: 0 !important;
          padding: 0.4rem 1.25rem !important;
        }
        .no-scroll-viewport .brand-name {
          font-size: 1.5rem !important;
          font-weight: 800 !important;
          letter-spacing: -0.02em !important;
        }
        .no-scroll-viewport .page-content {
          flex: 1 !important;
          display: flex !important;
          flex-direction: column !important;
          padding: 0.65rem 1.5rem !important;
          gap: 0.5rem !important;
          max-width: 1600px !important;
          width: 100% !important;
          margin: 0 auto !important;
          overflow: hidden !important;
          box-sizing: border-box !important;
        }
        .no-scroll-viewport .page-intro {
          flex-shrink: 0 !important;
          margin: 0 !important;
          padding: 0 !important;
          display: flex !important;
          align-items: center !important;
          justify-content: space-between !important;
        }
        .no-scroll-viewport .page-intro h1 {
          font-size: 2.35rem !important;
          font-weight: 800 !important;
          line-height: 1.05 !important;
          letter-spacing: -0.03em !important;
          margin: 0.1rem 0 !important;
        }
        .no-scroll-viewport .page-description {
          font-size: 0.85rem !important;
          margin: 0 !important;
        }
        .no-scroll-viewport .workspace-grid {
          flex: 1 !important;
          display: grid !important;
          grid-template-columns: 330px 1fr !important;
          gap: 1rem !important;
          min-height: 0 !important;
          overflow: hidden !important;
          align-items: stretch !important;
        }
        .no-scroll-viewport .form-panel {
          padding: 0.75rem 1rem !important;
          display: flex !important;
          flex-direction: column !important;
          justify-content: space-between !important;
          height: 100% !important;
          box-sizing: border-box !important;
          overflow: hidden !important;
        }
        .no-scroll-viewport .panel-heading {
          margin-bottom: 0.4rem !important;
          padding-bottom: 0.4rem !important;
        }
        .no-scroll-viewport .panel-heading h2 {
          font-size: 1rem !important;
          margin: 0 !important;
        }
        .no-scroll-viewport .panel-heading p {
          font-size: 0.75rem !important;
          margin: 0 !important;
        }
        .no-scroll-viewport .settings-form {
          display: flex !important;
          flex-direction: column !important;
          justify-content: space-between !important;
          flex: 1 !important;
          gap: 0.45rem !important;
          min-height: 0 !important;
          overflow: hidden !important;
        }
        .no-scroll-viewport .settings-section {
          padding: 0.45rem 0.6rem !important;
          margin: 0 !important;
          gap: 0.3rem !important;
        }
        .no-scroll-viewport .settings-section legend {
          font-size: 0.78rem !important;
          margin-bottom: 0.2rem !important;
        }
        .no-scroll-viewport .select-wrap select,
        .no-scroll-viewport .upload-control {
          padding: 0.3rem 0.5rem !important;
          font-size: 0.78rem !important;
          min-height: 30px !important;
        }
        .no-scroll-viewport .weight-options {
          gap: 0.35rem !important;
        }
        .no-scroll-viewport .weight-option {
          padding: 0.35rem 0.45rem !important;
          font-size: 0.75rem !important;
          min-height: 32px !important;
        }
        .no-scroll-viewport .capacity-field {
          padding: 0.2rem 0 !important;
          margin: 0 !important;
          min-height: 28px !important;
        }
        .no-scroll-viewport .capacity-field label {
          font-size: 0.78rem !important;
        }
        .no-scroll-viewport .capacity-input input {
          height: 26px !important;
          padding: 0.15rem 0.35rem !important;
          font-size: 0.8rem !important;
        }
        .no-scroll-viewport .find-routes-button {
          height: 36px !important;
          margin-top: 0.15rem !important;
        }
        .no-scroll-viewport .submit-hint {
          font-size: 0.7rem !important;
          margin: 0 !important;
        }
        .no-scroll-viewport .results-panel {
          padding: 0.85rem 1.15rem !important;
          display: flex !important;
          flex-direction: column !important;
          justify-content: space-between !important;
          gap: 0.55rem !important;
          height: 100% !important;
          min-height: 0 !important;
          box-sizing: border-box !important;
          overflow: hidden !important;
        }
        .no-scroll-viewport .results-heading {
          margin: 0 !important;
          padding-bottom: 0.35rem !important;
          flex-shrink: 0 !important;
        }
        .no-scroll-viewport .results-heading h2 {
          font-size: 1.25rem !important;
          margin: 0 !important;
        }
        .no-scroll-viewport .results-intro {
          font-size: 0.78rem !important;
          margin: 0 !important;
        }
        .no-scroll-viewport .overview-grid {
          gap: 0.65rem !important;
          margin: 0 !important;
          flex-shrink: 0 !important;
        }
        .no-scroll-viewport .overview-card {
          padding: 0.55rem 0.85rem !important;
        }
        .no-scroll-viewport .overview-value {
          font-size: 1.55rem !important;
          margin: 0.1rem 0 !important;
        }
        .no-scroll-viewport .overview-foot {
          font-size: 0.7rem !important;
          margin: 0 !important;
        }
        .no-scroll-viewport .results-columns {
          flex: 1 !important;
          min-height: 0 !important;
          gap: 0.75rem !important;
          margin: 0 !important;
          display: grid !important;
          grid-template-columns: 1fr 1fr !important;
          overflow: hidden !important;
        }
        .no-scroll-viewport .data-section {
          padding: 0.65rem 0.85rem !important;
          display: flex !important;
          flex-direction: column !important;
          min-height: 0 !important;
          height: 100% !important;
          box-sizing: border-box !important;
          overflow: hidden !important;
        }
        .no-scroll-viewport .section-heading {
          margin-bottom: 0.35rem !important;
          flex-shrink: 0 !important;
        }
        .no-scroll-viewport .section-heading h3 {
          font-size: 0.95rem !important;
          margin: 0 !important;
        }
        .no-scroll-viewport .route-donut {
          width: 130px !important;
          height: 130px !important;
          margin: 0.1rem auto !important;
          flex-shrink: 0 !important;
        }
        .no-scroll-viewport .donut-center strong {
          font-size: 1.05rem !important;
        }
        /* ONLY members inside the pie chart components scroll */
        .no-scroll-viewport .route-legend {
          flex: 1 !important;
          min-height: 0 !important;
          overflow-y: auto !important;
          padding-right: 0.25rem !important;
          gap: 0.25rem !important;
          margin-top: 0.35rem !important;
        }
        .no-scroll-viewport .composition-list {
          flex: 1 !important;
          min-height: 0 !important;
          overflow-y: auto !important;
          padding-right: 0.25rem !important;
          gap: 0.35rem !important;
        }
        .no-scroll-viewport .route-legend::-webkit-scrollbar,
        .no-scroll-viewport .composition-list::-webkit-scrollbar {
          width: 5px;
        }
        .no-scroll-viewport .route-legend::-webkit-scrollbar-track,
        .no-scroll-viewport .composition-list::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.04);
          border-radius: 9999px;
        }
        .no-scroll-viewport .route-legend::-webkit-scrollbar-thumb,
        .no-scroll-viewport .composition-list::-webkit-scrollbar-thumb {
          background: rgba(15, 23, 42, 0.22);
          border-radius: 9999px;
        }
        .no-scroll-viewport .route-row {
          padding: 0.2rem 0.35rem !important;
          font-size: 0.76rem !important;
        }
        .no-scroll-viewport .composition-row {
          gap: 0.15rem !important;
        }
        .no-scroll-viewport .composition-meta {
          font-size: 0.76rem !important;
        }
        .no-scroll-viewport .carbon-section {
          padding: 0.55rem 0.9rem !important;
          margin: 0 !important;
          flex-shrink: 0 !important;
        }
        .no-scroll-viewport .carbon-copy h3 {
          font-size: 0.95rem !important;
          margin: 0 !important;
        }
        .no-scroll-viewport .carbon-copy p {
          font-size: 0.75rem !important;
          margin: 0 !important;
        }
        .no-scroll-viewport .carbon-stat strong {
          font-size: 1.4rem !important;
        }
        .no-scroll-viewport .composition-note {
          font-size: 0.68rem !important;
          margin-top: 0.35rem !important;
          padding: 0 !important;
          flex-shrink: 0 !important;
        }
        .no-scroll-viewport .page-footer {
          padding: 0.25rem 0 !important;
          margin: 0 !important;
          font-size: 0.7rem !important;
          flex-shrink: 0 !important;
        }
      `}</style>

      {/* Header */}
      <header className="app-header">
        <div className="app-header-inner">
          <a className="brand-lockup" href="/" aria-label="PolyRoute home">
            <span className="brand-mark">
              <Recycle aria-hidden="true" />
            </span>
            <span className="brand-name">PolyRoute</span>
          </a>
          <div className="header-status">
            <span
              className="status-dot"
              style={{
                backgroundColor: isLive ? "#10b981" : "#f59e0b",
              }}
            />
            <span>ROUTE PLANNER</span>
            <span className="header-divider" />
            <span>{isLive ? "LIVE BACKEND" : "DEMO WORKSPACE"}</span>
          </div>
        </div>
      </header>

      {/* Main Single-Screen Content */}
      <main className="page-content">
        <div className="page-intro">
          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              gap: "1rem",
              flexWrap: "wrap",
            }}
          >
            <h1>PolyRoute</h1>
            <p className="page-description">
              Material Recovery Workspace — Find a clearer path for every
              plastic in your batch.
            </p>
          </div>
          <div className="intro-chip">
            <Sparkles aria-hidden="true" />{" "}
            {isLive ? `Connected: ${data.source}` : "Illustrative preview"}
          </div>
        </div>

        <div className="workspace-grid">
          {/* Left Aside: Batch & Facility Settings */}
          <aside
            className="form-panel"
            aria-label="Batch and facility settings"
          >
            <div className="panel-heading">
              <div className="panel-heading-icon">
                <SlidersHorizontal aria-hidden="true" />
              </div>
              <div>
                <h2>Batch settings</h2>
                <p>Set up a sample run</p>
              </div>
            </div>

            <form className="settings-form" onSubmit={runAnalysis}>
              {/* 01 Classifier output */}
              <fieldset className="settings-section">
                <legend>
                  <span className="step-number">01</span> Classifier output
                </legend>
                <div className="select-wrap">
                  <FileBraces aria-hidden="true" />
                  <select
                    id="sample-file"
                    value={sampleFile}
                    onChange={(e) => {
                      setSampleFile(e.target.value);
                      setClassifierFile(null);
                    }}
                  >
                    <option value="demo-mix">
                      Illustrative plastic mix - 100 items
                    </option>
                    {samples.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </div>

                <label className="upload-control" htmlFor="classifier-file">
                  <Upload aria-hidden="true" />
                  <span className="upload-name">
                    {classifierFile
                      ? classifierFile.name
                      : "Or upload JSON file"}
                  </span>
                  <span className="upload-action">Browse</span>
                </label>
                <input
                  id="classifier-file"
                  className="visually-hidden"
                  type="file"
                  accept=".json,application/json"
                  onChange={(e) =>
                    setClassifierFile(e.target.files?.[0] || null)
                  }
                />
              </fieldset>

              {/* 02 Batch weight */}
              <fieldset className="settings-section weight-section">
                <legend>
                  <span className="step-number">02</span> Batch weight
                </legend>
                <div
                  className="weight-options"
                  role="radiogroup"
                  aria-label="How to estimate batch weight"
                >
                  {(
                    [
                      { id: "measured", label: "Weighed", Icon: Scale },
                      { id: "volume", label: "By volume", Icon: Box },
                      { id: "count", label: "Item count", Icon: Hash },
                      { id: "photos", label: "Photos", Icon: Camera },
                    ] as const
                  ).map(({ id, label, Icon }) => {
                    const selected = weightMode === id;
                    return (
                      <label
                        key={id}
                        className={`weight-option ${selected ? "selected" : ""}`}
                        onClick={() => setWeightMode(id)}
                      >
                        <input
                          type="radio"
                          name="weight-mode"
                          value={id}
                          checked={selected}
                          onChange={() => setWeightMode(id)}
                        />
                        <Icon aria-hidden="true" />
                        <span>{label}</span>
                        {selected && (
                          <Check className="option-check" aria-hidden="true" />
                        )}
                      </label>
                    );
                  })}
                </div>

                {weightMode === "measured" && (
                  <div className="capacity-field">
                    <label htmlFor="batch-kg-input">
                      <Scale aria-hidden="true" />
                      <span>Total weight</span>
                    </label>
                    <div className="number-with-unit capacity-input">
                      <input
                        id="batch-kg-input"
                        type="number"
                        min="0.1"
                        step="0.5"
                        value={totalKgInput}
                        onChange={(e) => setTotalKgInput(e.target.value)}
                      />
                      <span>kg</span>
                    </div>
                  </div>
                )}

                {weightMode === "volume" && (
                  <div className="capacity-field">
                    <label htmlFor="batch-vol-input">
                      <Box aria-hidden="true" />
                      <span>Volume</span>
                    </label>
                    <div className="number-with-unit capacity-input">
                      <input
                        id="batch-vol-input"
                        type="number"
                        min="0.1"
                        step="0.1"
                        value={volumeM3Input}
                        onChange={(e) => setVolumeM3Input(e.target.value)}
                      />
                      <span>m³</span>
                    </div>
                  </div>
                )}

                {weightMode === "count" && (
                  <div className="capacity-field">
                    <label htmlFor="batch-count-input">
                      <Hash aria-hidden="true" />
                      <span>Total items</span>
                    </label>
                    <div className="number-with-unit capacity-input">
                      <input
                        id="batch-count-input"
                        type="number"
                        min="1"
                        step="1"
                        value={itemCountInput}
                        onChange={(e) => setItemCountInput(e.target.value)}
                      />
                      <span>pcs</span>
                    </div>
                  </div>
                )}

                {weightMode === "photos" && (
                  <p
                    className="mode-note"
                    role="status"
                    style={{ fontSize: "0.7rem", margin: 0 }}
                  >
                    Available when classifier is connected.
                  </p>
                )}
              </fieldset>

              {/* 03 Facility limits */}
              <fieldset className="settings-section facility-section">
                <legend>
                  <span className="step-number">03</span> Facility limits
                </legend>

                <div className="capacity-field">
                  <label htmlFor="recycling-capacity">
                    <Recycle aria-hidden="true" />
                    <span>Recycling</span>
                  </label>
                  <div className="number-with-unit capacity-input">
                    <input
                      id="recycling-capacity"
                      type="number"
                      min="0"
                      step="1"
                      value={recycleCap}
                      onChange={(e) => setRecycleCap(e.target.value)}
                    />
                    <span>kg</span>
                  </div>
                </div>

                <div className="capacity-field">
                  <label htmlFor="compost-capacity">
                    <Leaf aria-hidden="true" />
                    <span>Composting</span>
                  </label>
                  <div className="number-with-unit capacity-input">
                    <input
                      id="compost-capacity"
                      type="number"
                      min="0"
                      step="1"
                      value={compostCap}
                      onChange={(e) => setCompostCap(e.target.value)}
                    />
                    <span>kg</span>
                  </div>
                </div>

                <div className="capacity-field">
                  <label htmlFor="landfill-capacity">
                    <Trash2 aria-hidden="true" />
                    <span>Landfill limit</span>
                  </label>
                  <div className="number-with-unit capacity-input">
                    <input
                      id="landfill-capacity"
                      type="number"
                      min="0"
                      step="1"
                      value={landfillCap}
                      onChange={(e) => setLandfillCap(e.target.value)}
                    />
                    <span>kg</span>
                  </div>
                </div>
              </fieldset>

              <button
                className="inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium cursor-pointer transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 disabled:cursor-not-allowed [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 bg-primary text-primary-foreground shadow hover:bg-primary/90 rounded-md px-8 find-routes-button"
                type="submit"
                disabled={loading || weightMode === "photos"}
              >
                {loading ? (
                  <>
                    <span>Optimizing...</span>
                    <Loader2 className="animate-spin" aria-hidden="true" />
                  </>
                ) : (
                  <>
                    <span>Find routes</span>
                    <ArrowRight aria-hidden="true" />
                  </>
                )}
              </button>

              <p className="submit-hint">
                <Info aria-hidden="true" />{" "}
                {error
                  ? error
                  : isLive
                  ? "Connected to live Python optimizer."
                  : "Preview uses illustrative demo values."}
              </p>
            </form>
          </aside>

          {/* Right Section: Results Panel */}
          <section
            className="results-panel"
            aria-live="polite"
            aria-label="Route preview"
          >
            <div className="results-heading">
              <div
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  gap: "0.75rem",
                }}
              >
                <h2>Where your plastics go</h2>
                <p className="results-intro">
                  Route-by-route view of this {startingBatchKg.toFixed(1)} kg
                  batch
                </p>
              </div>
              <div className="preview-badge">
                <span />{" "}
                {isLive ? `${data.n_items} items analyzed` : "Illustrative data"}
              </div>
            </div>

            {/* 3 Overview Cards */}
            <div className="overview-grid">
              <article className="overview-card total-card">
                <div className="overview-label">
                  <Box aria-hidden="true" /> Batch analyzed
                </div>
                <div className="overview-value">
                  {startingBatchKg.toFixed(1)}
                  <span>kg</span>
                </div>
                <p className="overview-foot">Starting batch weight</p>
              </article>

              <article className="overview-card plastic-card">
                <div className="overview-label">
                  <Recycle aria-hidden="true" /> Plastic identified
                </div>
                <div className="overview-value">
                  {totalPlasticKg.toFixed(1)}
                  <span>kg</span>
                </div>
                <p className="overview-foot">
                  Across {data.lanes.length} material groups
                </p>
              </article>

              <article className="overview-card impact-card">
                <div className="overview-label">
                  <Leaf aria-hidden="true" /> Potential CO₂e avoided
                </div>
                <div className="overview-value">
                  {data.saved.toFixed(1)}
                  <span>kg</span>
                </div>
                <p className="overview-foot">Estimate vs. landfill</p>
              </article>
            </div>

            {/* Two-Column Breakdown: Route Donut + Composition Pie & Scrollable Member Lists */}
            <div className="results-columns">
              {/* Column 1: Route Breakdown */}
              <section
                className="data-section route-section"
                aria-labelledby="route-breakdown-title"
              >
                <div className="section-heading">
                  <div>
                    <p className="section-kicker">DESTINATION MIX</p>
                    <h3 id="route-breakdown-title">Route breakdown</h3>
                  </div>
                  <span className="unit-label">kg of plastic</span>
                </div>

                {/* Fixed Pie Chart */}
                <div
                  className="route-donut"
                  role="img"
                  aria-label="Route distribution pie chart"
                  style={{
                    ...conicShareStyles,
                    position: "relative",
                    background: "none",
                  }}
                >
                  <div
                    style={{
                      width: "100%",
                      height: "100%",
                      position: "absolute",
                      inset: 0,
                    }}
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={routePieData.filter((d) => d.value > 0)}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius="62%"
                          outerRadius="92%"
                          paddingAngle={2}
                          onMouseEnter={(_, idx) => setActiveRouteIndex(idx)}
                          onMouseLeave={() => setActiveRouteIndex(null)}
                        >
                          {routePieData
                            .filter((d) => d.value > 0)
                            .map((entry, idx) => (
                              <Cell
                                key={entry.key}
                                fill={entry.color}
                                stroke="#ffffff"
                                strokeWidth={2}
                                style={{
                                  opacity:
                                    activeRouteIndex === null ||
                                    activeRouteIndex === idx
                                      ? 1
                                      : 0.45,
                                  transition: "opacity 0.2s ease",
                                  cursor: "pointer",
                                }}
                              />
                            ))}
                        </Pie>
                        <RechartsTooltip
                          formatter={(val: number, name: string) => [
                            `${val.toFixed(1)} kg`,
                            name,
                          ]}
                          contentStyle={{
                            borderRadius: "8px",
                            border: "1px solid #e2e8f0",
                            fontSize: "11px",
                            padding: "4px 8px",
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <div
                    className="donut-center"
                    style={{ pointerEvents: "none", zIndex: 2 }}
                  >
                    <span style={{ fontSize: "0.65rem" }}>
                      {activeRouteIndex !== null &&
                      routePieData.filter((d) => d.value > 0)[activeRouteIndex]
                        ? routePieData.filter((d) => d.value > 0)[
                            activeRouteIndex
                          ].name
                        : "Total plastic"}
                    </span>
                    <strong>
                      {activeRouteIndex !== null &&
                      routePieData.filter((d) => d.value > 0)[activeRouteIndex]
                        ? routePieData
                            .filter((d) => d.value > 0)
                            [activeRouteIndex].value.toFixed(1)
                        : totalPlasticKg.toFixed(1)}
                      <small>kg</small>
                    </strong>
                  </div>
                </div>

                {/* Scrollable Route Members */}
                <div className="route-legend">
                  {routePieData.map((r, idx) => (
                    <div
                      className="route-row"
                      key={r.key}
                      onMouseEnter={() => setActiveRouteIndex(idx)}
                      onMouseLeave={() => setActiveRouteIndex(null)}
                      style={{ cursor: "pointer" }}
                    >
                      <span
                        className={`route-dot ${r.dotClass}`}
                        style={{ backgroundColor: r.color }}
                      />
                      <span className="route-name">{r.name}</span>
                      <span className="route-weight">
                        {r.value.toFixed(1)}kg
                      </span>
                      <span className="route-percent">{r.pct}%</span>
                    </div>
                  ))}
                </div>
              </section>

              {/* Column 2: Material Profile */}
              <section
                className="data-section composition-section"
                aria-labelledby="composition-title"
              >
                <div className="section-heading">
                  <div>
                    <p className="section-kicker">MATERIAL PROFILE</p>
                    <h3 id="composition-title">What's in the batch</h3>
                  </div>
                  <span className="unit-label">
                    {isLive ? "classified mix" : "illustrative mix"}
                  </span>
                </div>

                {/* Fixed Polymer Pie Chart */}
                <div
                  style={{
                    width: "130px",
                    height: "130px",
                    margin: "0.1rem auto",
                    position: "relative",
                    flexShrink: 0,
                  }}
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={polymerPieData}
                        dataKey="value"
                        nameKey="polymer"
                        cx="50%"
                        cy="50%"
                        innerRadius="55%"
                        outerRadius="92%"
                        paddingAngle={3}
                        onMouseEnter={(_, idx) => setActivePolymerIndex(idx)}
                        onMouseLeave={() => setActivePolymerIndex(null)}
                      >
                        {polymerPieData.map((entry, idx) => (
                          <Cell
                            key={entry.polymer}
                            fill={entry.color}
                            stroke="#ffffff"
                            strokeWidth={2}
                            style={{
                              opacity:
                                activePolymerIndex === null ||
                                activePolymerIndex === idx
                                  ? 1
                                  : 0.45,
                              transition: "opacity 0.2s ease",
                              cursor: "pointer",
                            }}
                          />
                        ))}
                      </Pie>
                      <RechartsTooltip
                        formatter={(val: number, name: string, props: any) => [
                          `${val.toFixed(1)} kg (${props.payload.pct}%)`,
                          name,
                        ]}
                        contentStyle={{
                          borderRadius: "8px",
                          border: "1px solid #e2e8f0",
                          fontSize: "11px",
                          padding: "4px 8px",
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                {/* Scrollable Polymer Members */}
                <div className="composition-list">
                  {polymerPieData.map((item, idx) => (
                    <div
                      className="composition-row"
                      key={item.polymer}
                      onMouseEnter={() => setActivePolymerIndex(idx)}
                      onMouseLeave={() => setActivePolymerIndex(null)}
                      style={{
                        cursor: "pointer",
                        opacity:
                          activePolymerIndex === null ||
                          activePolymerIndex === idx
                            ? 1
                            : 0.55,
                        transition: "opacity 0.2s ease",
                      }}
                    >
                      <div className="composition-meta">
                        <span
                          className={`material-dot material-dot-${idx % 4}`}
                          style={{ backgroundColor: item.color }}
                        />
                        <span className="material-name" title={item.name}>
                          {item.polymer}
                        </span>
                        <span className="material-weight">
                          {item.value.toFixed(1)}kg
                        </span>
                        <span className="material-share">{item.pct}%</span>
                      </div>
                      <div className="composition-track" aria-hidden="true">
                        <span
                          className={`composition-fill composition-fill-${idx % 4}`}
                          style={{
                            width: `${item.pct}%`,
                            backgroundColor: item.color,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="composition-note">
                  <Info aria-hidden="true" />{" "}
                  {isLive
                    ? "Hover any polymer slice or row to inspect its share."
                    : "Polymer shares are sample values."}
                </div>
              </section>
            </div>

            {/* Climate Comparison Section */}
            <section className="carbon-section" aria-labelledby="carbon-title">
              <div className="carbon-icon">
                <Leaf aria-hidden="true" />
              </div>
              <div className="carbon-copy">
                <p className="section-kicker">CLIMATE COMPARISON</p>
                <h3 id="carbon-title">A lower-carbon path, at a glance</h3>
                <p>
                  {data.explanation
                    ? data.explanation
                    : "Example estimate against sending the identified plastic to landfill."}
                </p>
              </div>
              <div className="carbon-stat">
                <strong>{data.saved.toFixed(1)}</strong>
                <span>kg CO₂e avoided*</span>
              </div>
              <div className="carbon-decoration" aria-hidden="true">
                <Factory aria-hidden="true" />
                <ArrowUpRight aria-hidden="true" />
              </div>
            </section>
          </section>
        </div>

        {/* Footer */}
        <footer className="page-footer">
          <span>POLYROUTE</span>
          <span>Materials in motion, with less waste.</span>
        </footer>
      </main>
    </div>
  );
}

export default PolyRouteApp;
