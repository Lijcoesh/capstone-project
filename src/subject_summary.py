#!/usr/bin/env python3
"""
subject_summary.py

Leest alle events.tsv bestanden van een SeizIT2 subject (alle sessies/runs)
en schrijft een uitgebreide, leesbare samenvatting naar een .txt bestand.

Gebruik (één subject, zoals voorheen):
    python subject_summary.py --subject 043
    python subject_summary.py --subject 043 --outdir results/subject_summaries

Gebruik (ALLE subjects -> één samenvattende CSV, voor cross-subject analyse):
    python subject_summary.py --all
    python subject_summary.py --all --csv-out results/all_subjects_summary.csv
    python subject_summary.py --all --no-txt   # alleen CSV, geen losse .txt per subject

Verwachte mappenstructuur (BIDS):
    <SEIZEIT2_BASE>/sub-<NNN>/ses-<NN>/eeg/sub-<NNN>_ses-<NN>_task-szMonitoring_run-<NN>_events.tsv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd

# Pas dit pad aan als jouw structuur afwijkt
SEIZEIT2_BASE = Path(__file__).resolve().parent / "../data/raw/seizeit2"

# Probeer de echte preprocessing-module te importeren, zodat de summary de
# DAADWERKELIJK GECONFIGUREERDE waarden rapporteert (niet comments, niet aannames).
# Dit voorkomt precies het soort mismatch dat we eerder tegenkwamen (comment zei
# 10 min, PREICTAL_SEC stond op 30*60). Als de import faalt, draait het script
# gewoon door zonder die sectie (events.tsv-analyse blijft sowieso werken).
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import preprocess_common as ppc
    PPC_AVAILABLE = True
    PPC_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - we willen elke importfout tonen, niet crashen
    PPC_AVAILABLE = False
    PPC_IMPORT_ERROR = str(e)


# Echte seizure-events in SeizIT2 beginnen met 'sz_' (bv. sz_foc_ia, sz_foc_ia_m_tonic).
# 'impd' = impedance check, GEEN seizure en GEEN bruikbaar inter-ictaal segment —
# dit is een technisch kwaliteitsmarker en moet apart gehouden worden.
# Module-niveau, zodat build_summary() EN compute_subject_stats() dezelfde definitie delen.
NON_SEIZURE_TYPES = {"bckg", "background", "bgr"}
ARTEFACT_TYPES = {"impd", "impedance"}


def find_event_files(base: Path, subject: str) -> list[Path]:
    """Vind alle events.tsv bestanden voor een subject, over alle sessies/runs heen."""
    subject_dir = base / f"sub-{subject}"
    if not subject_dir.exists():
        return []
    # Zoek recursief, want sessie/run-aantal kan per subject verschillen
    files = sorted(subject_dir.glob("ses-*/eeg/sub-*_events.tsv"))
    return files


def find_all_subjects(base: Path) -> list[str]:
    """Vind alle subject-IDs (zonder 'sub-' prefix) onder de base map."""
    if not base.exists():
        return []
    subjects = []
    for d in sorted(base.glob("sub-*")):
        if d.is_dir():
            subjects.append(d.name.replace("sub-", "", 1))
    return subjects


def parse_events_file(path: Path) -> pd.DataFrame:
    """Lees één events.tsv in en voeg metadata over herkomst toe."""
    df = pd.read_csv(path, sep="\t")
    # Haal ses-/run-info uit de bestandsnaam voor traceerbaarheid
    name = path.stem  # zonder .tsv
    parts = {}
    for token in name.split("_"):
        if "-" in token:
            key, _, val = token.partition("-")
            parts[key] = val
    df["source_session"] = parts.get("ses", "n/a")
    df["source_run"] = parts.get("run", "n/a")
    df["source_file"] = path.name
    return df


def safe_mean(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.mean() if len(s) else None


def format_value_counts(series: pd.Series, label: str) -> list[str]:
    lines = [f"  {label}:"]
    counts = series.fillna("n/a").value_counts(dropna=False)
    if counts.empty:
        lines.append("    (geen data)")
        return lines
    for val, count in counts.items():
        lines.append(f"    {val}: {count}")
    return lines


def build_summary(subject: str, all_events: pd.DataFrame, files: list[Path]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"SUBJECT SUMMARY — sub-{subject}")
    lines.append(f"Gegenereerd op: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 70)
    lines.append("")

    # --- Overzicht bronbestanden ---
    lines.append(f"Aantal gevonden events.tsv bestanden: {len(files)}")
    for f in files:
        lines.append(f"  - {f.relative_to(f.parents[3])}" if len(f.parents) >= 4 else f"  - {f}")
    lines.append("")

    if all_events.empty:
        lines.append("GEEN EVENTS GEVONDEN. Controleer pad en subject-nummer.")
        return "\n".join(lines)

    # --- Recording-niveau statistieken ---
    lines.append("-" * 70)
    lines.append("RECORDINGS")
    lines.append("-" * 70)
    rec_groups = all_events.groupby(["source_session", "source_run"])
    total_recording_seconds = 0.0
    for (ses, run), group in rec_groups:
        rec_dur = pd.to_numeric(group["recordingDuration"], errors="coerce").dropna()
        dur = rec_dur.iloc[0] if len(rec_dur) else None
        if dur is not None:
            total_recording_seconds += dur
        lines.append(
            f"  ses-{ses} run-{run}: recordingDuration = "
            f"{dur if dur is not None else 'n/a'} s "
            f"({dur/3600:.2f} u)" if dur is not None else
            f"  ses-{ses} run-{run}: recordingDuration = n/a"
        )
    lines.append(f"  TOTALE opnameduur over alle runs: {total_recording_seconds:.2f} s "
                  f"({total_recording_seconds/3600:.2f} uur)")
    lines.append("")

    # --- Event type overzicht ---
    lines.append("-" * 70)
    lines.append("EVENT TYPES (alle rijen, inclusief achtergrond)")
    lines.append("-" * 70)
    lines.extend(format_value_counts(all_events["eventType"], "eventType counts"))
    lines.append("")

    # --- Categoriseer eventType in drie groepen: seizure / background / artefact ---
    # (NON_SEIZURE_TYPES / ARTEFACT_TYPES zijn module-level gedefinieerd, gedeeld met
    #  compute_subject_stats() zodat .txt en .csv nooit uit elkaar kunnen lopen)
    event_type_lower = all_events["eventType"].astype(str).str.lower()
    seizure_mask = event_type_lower.str.startswith("sz_")
    artefact_mask = event_type_lower.isin(ARTEFACT_TYPES)
    bckg_mask = event_type_lower.isin(NON_SEIZURE_TYPES)
    other_mask = ~seizure_mask & ~artefact_mask & ~bckg_mask

    seizures = all_events[seizure_mask].copy()
    artefacts = all_events[artefact_mask].copy()
    other_events = all_events[other_mask].copy()

    lines.append("-" * 70)
    lines.append("SEIZURE EVENTS — DETAILS")
    lines.append("-" * 70)
    lines.append(f"  Totaal aantal seizure-events: {len(seizures)}")
    lines.append("")

    if len(seizures) == 0:
        lines.append("  Geen seizure-events gevonden voor deze subject (mogelijk seizure-vrije subject).")
    else:
        # Duur-statistieken
        durations = pd.to_numeric(seizures["duration"], errors="coerce").dropna()
        if len(durations):
            lines.append("  Duur (seconden):")
            lines.append(f"    min:    {durations.min():.2f}")
            lines.append(f"    max:    {durations.max():.2f}")
            lines.append(f"    mean:   {durations.mean():.2f}")
            lines.append(f"    median: {durations.median():.2f}")
        lines.append("")

        # Categorische velden
        for col, label in [
            ("eventType", "eventType (seizure-rijen)"),
            ("lateralization", "lateralization"),
            ("localization", "localization"),
            ("vigilance", "vigilance"),
            ("confidence", "confidence"),
        ]:
            if col in seizures.columns:
                lines.extend(format_value_counts(seizures[col], label))
                lines.append("")

        # Per-event volledige rij, voor wie het precies wil nazien
        lines.append("  Individuele seizure-events (volledige rij per event):")
        display_cols = [c for c in seizures.columns if c not in ("source_file",)]
        for idx, row in seizures.reset_index(drop=True).iterrows():
            lines.append(f"    [{idx}] ses-{row.get('source_session','n/a')} "
                         f"run-{row.get('source_run','n/a')}")
            for col in display_cols:
                if col in ("source_session", "source_run"):
                    continue
                lines.append(f"        {col}: {row[col]}")
        lines.append("")

    # --- Artefact / technische events (bv. impedance checks) ---
    lines.append("-" * 70)
    lines.append("ARTEFACT / TECHNISCHE EVENTS (bv. impedance) — NIET seizure, NIET bckg")
    lines.append("-" * 70)
    lines.append(f"  Totaal aantal: {len(artefacts)}")
    if len(artefacts):
        art_durations = pd.to_numeric(artefacts["duration"], errors="coerce")
        lines.append(f"  Duur min/max/mean: {art_durations.min():.2f} / "
                      f"{art_durations.max():.2f} / {art_durations.mean():.2f} s")
        n_negative = (art_durations < 0).sum()
        if n_negative:
            lines.append(f"  WAARSCHUWING: {n_negative} artefact-rij(en) met NEGATIEVE duration — "
                          f"check brondata, dit is fysiek onmogelijk.")
        lines.extend(format_value_counts(artefacts["eventType"], "eventType (artefact-rijen)"))
    lines.append("")
    lines.append("  LET OP voor je training-pipeline: 'impd' segmenten zijn impedance-checks, geen")
    lines.append("  bruikbaar inter-ictaal signaal. Controleer of deze in je preprocessing apart")
    lines.append("  worden gehouden en niet als 'bckg'/inter-ictaal worden gelabeld.")
    lines.append("")

    if len(other_events):
        lines.append("-" * 70)
        lines.append("OVERIGE / ONHERKENDE EVENTTYPES")
        lines.append("-" * 70)
        lines.extend(format_value_counts(other_events["eventType"], "eventType (overig)"))
        lines.append("")

    # --- Inter-ictaal (bckg) overzicht ---
    bckg = all_events[bckg_mask]
    lines.append("-" * 70)
    lines.append("BACKGROUND (INTER-ICTAAL) OVERZICHT")
    lines.append("-" * 70)
    bckg_durations = pd.to_numeric(bckg["duration"], errors="coerce").dropna()
    if len(bckg_durations):
        lines.append(f"  Aantal bckg-segmenten: {len(bckg)}")
        lines.append(f"  Totale bckg-duur: {bckg_durations.sum():.2f} s "
                      f"({bckg_durations.sum()/3600:.2f} uur)")
    else:
        lines.append("  Geen bckg-segmenten gevonden.")
    lines.append("")

    # --- Samenvattend kengetal: seizures per uur opname ---
    lines.append("-" * 70)
    lines.append("KERNGETALLEN")
    lines.append("-" * 70)
    if total_recording_seconds > 0:
        rate = len(seizures) / (total_recording_seconds / 3600)
        lines.append(f"  Seizures per uur opname: {rate:.4f}")
    lines.append(f"  Aantal runs/sessies: {len(rec_groups)}")
    lines.append("")

    # --- Preprocessing-pipeline check: gebruikt de DAADWERKELIJKE waarden uit
    #     preprocess_common.py, niet aannames. Simuleert hoeveel windows deze
    #     subject zou opleveren met de huidige config.
    lines.append("-" * 70)
    lines.append("PREPROCESSING PIPELINE CHECK (live import van preprocess_common.py)")
    lines.append("-" * 70)
    if not PPC_AVAILABLE:
        lines.append(f"  KON preprocess_common.py NIET IMPORTEREN: {PPC_IMPORT_ERROR}")
        lines.append("  (Zorg dat preprocess_common.py in dezelfde map staat als dit script,")
        lines.append("   of pas het sys.path.insert(...) bovenin dit script aan.)")
    else:
        lines.append(f"  PREICTAL_SEC          = {ppc.PREICTAL_SEC}  "
                      f"({ppc.PREICTAL_SEC / 60:.1f} min)")
        lines.append(f"  POSTICTAL_GUARD_SEC   = {ppc.POSTICTAL_GUARD_SEC}  "
                      f"({ppc.POSTICTAL_GUARD_SEC / 60:.1f} min)")
        lines.append(f"  DEFAULT_LABEL_MODE    = {ppc.DEFAULT_LABEL_MODE}")
        lines.append(f"  DEFAULT_WINDOW_SEC    = {ppc.DEFAULT_WINDOW_SEC}")
        lines.append(f"  DEFAULT_STEP_SEC      = {ppc.DEFAULT_STEP_SEC}")
        lines.append(f"  DEFAULT_INTERICTAL_RATIO = {ppc.DEFAULT_INTERICTAL_RATIO}")
        lines.append(f"  DEFAULT_NORMALIZE     = {ppc.DEFAULT_NORMALIZE}")
        lines.append(f"  DEFAULT_INPUT_REP     = {ppc.DEFAULT_INPUT_REP}")
        lines.append("")

        # parse_seizure_events gebruikt dezelfde 'sz' prefix-regel als hierboven,
        # maar dit roept de ECHTE functie uit de pipeline aan -> geen kans op drift
        # tussen wat dit script denkt en wat de training-code echt doet.
        sim_lines = []
        total_pos = 0
        total_inter = 0
        total_excl = 0
        for f in files:
            ppc_seizures = ppc.parse_seizure_events(f)
            row0 = pd.read_csv(f, sep="\t").iloc[0] if True else None
            rec_dur_series = pd.to_numeric(
                pd.read_csv(f, sep="\t")["recordingDuration"], errors="coerce"
            ).dropna()
            rec_dur = rec_dur_series.iloc[0] if len(rec_dur_series) else None
            if rec_dur is None or not ppc_seizures:
                sim_lines.append(f"  {f.name}: {len(ppc_seizures)} sz-event(en) via "
                                  f"parse_seizure_events, geen simulatie (geen seizures "
                                  f"of onbekende recordingDuration)")
                continue

            win = ppc.DEFAULT_WINDOW_SEC
            step = ppc.DEFAULT_STEP_SEC
            pos_n, inter_n, excl_n = 0, 0, 0
            ws = 0.0
            while ws + win <= rec_dur:
                label = ppc._classify_window(ws, ws + win, ppc_seizures,
                                              preictal_sec=ppc.PREICTAL_SEC,
                                              label_mode=ppc.DEFAULT_LABEL_MODE)
                if label == "positive":
                    pos_n += 1
                elif label == "interictal":
                    inter_n += 1
                else:
                    excl_n += 1
                ws += step
            total_pos += pos_n
            total_inter += inter_n
            total_excl += excl_n
            sim_lines.append(f"  {f.name}:")
            sim_lines.append(f"      sz-events (via parse_seizure_events): {len(ppc_seizures)}")
            sim_lines.append(f"      windows -> positive={pos_n}  interictal={inter_n}  "
                              f"excluded={excl_n}")

        lines.append("  Simulatie van window-labeling MET DE ECHTE PIPELINE-FUNCTIES")
        lines.append("  (window_sec/step_sec = de DEFAULT_* hierboven; als je build_dataset")
        lines.append("   met andere waarden aanroept, wijkt dit af van je echte run):")
        lines.extend(sim_lines)
        lines.append("")
        lines.append(f"  TOTAAL over alle runs: positive={total_pos}  interictal={total_inter}  "
                      f"excluded={total_excl}")
        if total_pos > 0:
            lines.append(f"  Effectieve interictal:positive ratio vóór subsampling: "
                          f"{total_inter / total_pos:.2f} : 1")
            target_n = int(round(ppc.DEFAULT_INTERICTAL_RATIO * total_pos))
            kept_n = min(target_n, total_inter)
            if target_n > total_inter:
                lines.append(f"  DEFAULT_INTERICTAL_RATIO={ppc.DEFAULT_INTERICTAL_RATIO} zou "
                              f"{target_n} interictal windows willen, maar er zijn er maar "
                              f"{total_inter} beschikbaar -> ALLE {total_inter} worden behouden "
                              f"(ratio komt dus lager uit dan ingesteld voor deze subject).")
            else:
                lines.append(f"  Na subsampling naar DEFAULT_INTERICTAL_RATIO="
                              f"{ppc.DEFAULT_INTERICTAL_RATIO}: {kept_n} interictal windows "
                              f"zouden behouden blijven (van de {total_inter} beschikbare).")
        else:
            lines.append("  Geen positive windows -> deze subject draagt NIETS bij aan de")
            lines.append("  trainingset (build_recording_windows geeft lege array terug).")
    lines.append("")

    lines.append("=" * 70)
    lines.append("EINDE SAMENVATTING")
    lines.append("=" * 70)

    return "\n".join(lines)


def compute_subject_stats(subject: str, all_events: pd.DataFrame, files: list[Path]) -> dict:
    """
    Bereken kerncijfers voor één subject als platte dict — bedoeld voor de
    cross-subject CSV (--all). Hergebruikt dezelfde classificatielogica als
    build_summary(), zodat .txt en .csv nooit uit elkaar kunnen lopen.
    """
    row = {"subject": f"sub-{subject}"}

    if all_events.empty:
        row["error"] = "geen events gevonden"
        return row

    # --- recordings ---
    rec_groups = all_events.groupby(["source_session", "source_run"])
    total_recording_seconds = 0.0
    for (ses, run), group in rec_groups:
        rec_dur = pd.to_numeric(group["recordingDuration"], errors="coerce").dropna()
        if len(rec_dur):
            total_recording_seconds += rec_dur.iloc[0]
    row["n_runs"] = len(rec_groups)
    row["total_recording_hours"] = round(total_recording_seconds / 3600, 4)

    # --- categoriseren (zelfde regels als build_summary) ---
    event_type_lower = all_events["eventType"].astype(str).str.lower()
    seizure_mask = event_type_lower.str.startswith("sz_")
    artefact_mask = event_type_lower.isin(ARTEFACT_TYPES)
    bckg_mask = event_type_lower.isin(NON_SEIZURE_TYPES)

    seizures = all_events[seizure_mask].copy()
    artefacts = all_events[artefact_mask].copy()
    bckg = all_events[bckg_mask]

    # --- seizure stats ---
    row["n_seizures"] = len(seizures)
    durations = pd.to_numeric(seizures["duration"], errors="coerce").dropna()
    row["seizure_duration_min_s"] = round(durations.min(), 2) if len(durations) else None
    row["seizure_duration_max_s"] = round(durations.max(), 2) if len(durations) else None
    row["seizure_duration_mean_s"] = round(durations.mean(), 2) if len(durations) else None
    row["seizure_duration_median_s"] = round(durations.median(), 2) if len(durations) else None
    row["seizures_per_hour"] = (
        round(len(seizures) / (total_recording_seconds / 3600), 4)
        if total_recording_seconds > 0 else None
    )

    # meest voorkomende categorische waarden (handig om snel te scannen/filteren)
    for col in ("eventType", "lateralization", "localization", "vigilance"):
        if col in seizures.columns and len(seizures):
            vc = seizures[col].fillna("n/a").value_counts()
            row[f"seizure_{col}_top"] = vc.index[0]
            row[f"seizure_{col}_values"] = ";".join(f"{k}:{v}" for k, v in vc.items())
        else:
            row[f"seizure_{col}_top"] = None
            row[f"seizure_{col}_values"] = None

    # --- artefact (impd) stats ---
    row["n_artefacts"] = len(artefacts)
    art_durations = pd.to_numeric(artefacts["duration"], errors="coerce") if len(artefacts) else pd.Series(dtype=float)
    row["artefact_duration_min_s"] = round(art_durations.min(), 2) if len(art_durations) else None
    row["artefact_duration_max_s"] = round(art_durations.max(), 2) if len(art_durations) else None
    row["artefact_n_negative_duration"] = int((art_durations < 0).sum()) if len(art_durations) else 0

    # --- bckg stats ---
    bckg_durations = pd.to_numeric(bckg["duration"], errors="coerce").dropna()
    row["n_bckg_segments"] = len(bckg)
    row["bckg_total_hours"] = round(bckg_durations.sum() / 3600, 4) if len(bckg_durations) else 0.0

    # --- preprocessing-pipeline simulatie (alleen als ppc beschikbaar is) ---
    if PPC_AVAILABLE:
        total_pos, total_inter, total_excl = 0, 0, 0
        for f in files:
            ppc_seizures = ppc.parse_seizure_events(f)
            if not ppc_seizures:
                continue
            rec_dur_series = pd.to_numeric(
                pd.read_csv(f, sep="\t")["recordingDuration"], errors="coerce"
            ).dropna()
            rec_dur = rec_dur_series.iloc[0] if len(rec_dur_series) else None
            if rec_dur is None:
                continue
            win, step = ppc.DEFAULT_WINDOW_SEC, ppc.DEFAULT_STEP_SEC
            ws = 0.0
            while ws + win <= rec_dur:
                label = ppc._classify_window(ws, ws + win, ppc_seizures,
                                              preictal_sec=ppc.PREICTAL_SEC,
                                              label_mode=ppc.DEFAULT_LABEL_MODE)
                if label == "positive":
                    total_pos += 1
                elif label == "interictal":
                    total_inter += 1
                else:
                    total_excl += 1
                ws += step
        row["sim_positive_windows"] = total_pos
        row["sim_interictal_windows"] = total_inter
        row["sim_excluded_windows"] = total_excl
        row["sim_interictal_to_positive_ratio"] = (
            round(total_inter / total_pos, 2) if total_pos > 0 else None
        )
        target_n = int(round(ppc.DEFAULT_INTERICTAL_RATIO * total_pos)) if total_pos else 0
        row["sim_interictal_kept_after_subsample"] = min(target_n, total_inter) if total_pos else 0
        row["sim_ratio_target_unmet"] = bool(total_pos > 0 and target_n > total_inter)
    else:
        for k in ("sim_positive_windows", "sim_interictal_windows", "sim_excluded_windows",
                   "sim_interictal_to_positive_ratio", "sim_interictal_kept_after_subsample",
                   "sim_ratio_target_unmet"):
            row[k] = None

    return row


def load_subject_events(base: Path, subject: str) -> tuple[list[Path], pd.DataFrame | None]:
    """Vind en laad alle events.tsv voor een subject. Geeft (files, df) terug;
    df is None als er geen leesbare bestanden waren."""
    files = find_event_files(base, subject)
    if not files:
        return files, None
    all_dfs = []
    for f in files:
        try:
            all_dfs.append(parse_events_file(f))
        except Exception as e:
            print(f"  WAARSCHUWING: kon {f} niet lezen ({e})", file=sys.stderr)
    if not all_dfs:
        return files, None
    return files, pd.concat(all_dfs, ignore_index=True)


def process_subject(base: Path, subject: str, outdir: Path | None) -> dict:
    """Verwerk één subject: laad events, optioneel .txt wegschrijven, en geef
    altijd de CSV-rij (dict) terug zodat --all dit kan hergebruiken."""
    files, all_events = load_subject_events(base, subject)
    if not files:
        print(f"  WAARSCHUWING: geen events.tsv gevonden voor sub-{subject}", file=sys.stderr)
        return {"subject": f"sub-{subject}", "error": "geen events.tsv gevonden"}
    if all_events is None:
        return {"subject": f"sub-{subject}", "error": "geen enkel bestand leesbaar"}

    if outdir is not None:
        summary_text = build_summary(subject, all_events, files)
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"sub-{subject}_summary.txt"
        outpath.write_text(summary_text, encoding="utf-8")
        print(f"  .txt geschreven: {outpath}")

    return compute_subject_stats(subject, all_events, files)


def main():
    parser = argparse.ArgumentParser(description="SeizIT2 subject events.tsv samenvatter")
    parser.add_argument("--subject", help="Subject nummer, bijv. 001, 043, 125 (zonder 'sub-' prefix)")
    parser.add_argument("--all", action="store_true",
                         help="Verwerk ALLE subjects onder --base en schrijf één samenvattende CSV")
    parser.add_argument("--base", default=str(SEIZEIT2_BASE),
                         help="Pad naar de SeizIT2 root map (bevat sub-* mappen)")
    parser.add_argument("--outdir", default="capstone-project/results/subject_summaries",
                         help="Map waar .txt bestand(en) komen te staan")
    parser.add_argument("--csv-out", default="results/all_subjects_summary.csv",
                         help="Pad voor de samenvattende CSV (alleen gebruikt met --all)")
    parser.add_argument("--no-txt", action="store_true",
                         help="Bij --all: sla per-subject .txt bestanden over, schrijf alleen de CSV")
    args = parser.parse_args()

    if not args.subject and not args.all:
        parser.error("geef --subject <nummer> op, of gebruik --all voor alle subjects")

    base = Path(args.base)

    if args.all:
        subjects = find_all_subjects(base)
        if not subjects:
            print(f"FOUT: geen sub-* mappen gevonden onder {base}", file=sys.stderr)
            sys.exit(1)
        print(f"Gevonden {len(subjects)} subject(en) onder {base}")

        outdir = None if args.no_txt else Path(args.outdir)
        rows = []
        for subject in subjects:
            print(f"\n[sub-{subject}]")
            rows.append(process_subject(base, subject, outdir))

        df = pd.DataFrame(rows)
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

        n_errors = df["error"].notna().sum() if "error" in df.columns else 0
        print(f"\n{'=' * 70}")
        print(f"CSV weggeschreven: {csv_path.resolve()}")
        print(f"  {len(df)} subjects, {n_errors} met fouten/ontbrekende data")
        if not args.no_txt:
            print(f"  Per-subject .txt bestanden in: {outdir.resolve()}")
        print(f"{'=' * 70}")
        return

    # --- single-subject modus (zoals voorheen) ---
    subject = args.subject.zfill(3) if args.subject.isdigit() else args.subject
    row = process_subject(base, subject, Path(args.outdir))
    if "error" in row:
        print(f"FOUT: {row['error']} voor sub-{subject}", file=sys.stderr)
        sys.exit(1)
    print(f"\nSamenvatting weggeschreven naar: "
          f"{(Path(args.outdir) / f'sub-{subject}_summary.txt').resolve()}")


if __name__ == "__main__":
    main()