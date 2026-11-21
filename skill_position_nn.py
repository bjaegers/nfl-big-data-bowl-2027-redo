"""
Skill Position Combine & Movement Neural Network Projector
==========================================================
Dual-branch multi-task neural network (pure NumPy implementation with Adam,
Cosine Annealing, Huber/Smooth-L1 loss, Dropout, and Residual skip connections)
that projects:
  1. Rookie Year & Per-Game Yards (yards_per_game)
  2. Rookie Year & Per-Game Touchdowns (tds_per_game)
  3. 4th-Quarter / Clutch Scoring Rate (q4_tds_per_game)
  4. 4th-Quarter Clutch Yards & Explosive Rate (clutch_yards_per_game)
  5. 4th-Quarter Efficiency Step-Up (q4_vs_q13_ypr_diff) & Clutch WPA
  6. Game-by-Game Yards & Touchdowns conditioned on game opportunity & late-game state
"""

import os
import math
import duckdb
import numpy as np
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_data.duckdb")


# ---------------------------------------------------------------------------
# Statistical Helpers (Pure NumPy / Math)
# ---------------------------------------------------------------------------
def pearson_r_pval(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
    if denom == 0:
        return 0.0, 1.0
    r = float(np.clip(np.sum(xm * ym) / denom, -0.999999, 0.999999))
    df_dof = n - 2
    t_stat = abs(r) * math.sqrt(df_dof / (1.0 - r * r))
    grid = np.linspace(0, t_stat, 501)
    log_c = (
        math.lgamma((df_dof + 1) / 2.0)
        - 0.5 * math.log(df_dof * math.pi)
        - math.lgamma(df_dof / 2.0)
    )
    pdf = np.exp(log_c - ((df_dof + 1) / 2.0) * np.log(1.0 + (grid ** 2) / df_dof))
    cdf_half = np.trapezoid(pdf, grid)
    pval = max(0.0, min(1.0, 2.0 * (0.5 - cdf_half)))
    return r, pval


def spearman_rho_pval(x, y):
    s1 = pd.Series(x).rank()
    s2 = pd.Series(y).rank()
    return pearson_r_pval(s1.values, s2.values)


def _is_player_td(row):
    desc = str(row["play_description"])
    if "TOUCHDOWN" not in desc.upper():
        return 0
    upper_desc = desc.upper()
    if "REVERSED" in upper_desc:
        parts = upper_desc.split("REVERSED")
        if "TOUCHDOWN" not in parts[-1]:
            return 0
    rec = row["rec_yards"] if pd.notna(row["rec_yards"]) else 0
    rush = row["rush_yards"] if pd.notna(row["rush_yards"]) else 0
    fumble_lost = row["fumble_lost"] if pd.notna(row["fumble_lost"]) else 0
    if (rec > 0 or rush > 0) and fumble_lost == 0:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Feature Extraction Pipeline from DuckDB
# ---------------------------------------------------------------------------
def build_skill_position_dataset(con=None):
    """
    Extracts and engineers:
      - Combine physical measurements & athletic indices from `combine_results`
      - Frame-by-frame Combine drill movement kinematics & phase profiles from `combine_tracking`
      - Rookie-year (season == draft_year) and career per-game, per-route, and Q4/Clutch targets
      - Individual player-game logs (`player_games_df`) across all rookie and career games
    """
    close_con = False
    if con is None:
        con = duckdb.connect(DB_PATH, read_only=True)
        close_con = True

    # 1. Base skill-position players + combine_results
    players_df = con.execute("""
        SELECT 
            p.nfl_id, p.display_name, p.nfl_position, p.draft_year,
            p.college_name, p.college_conference,
            p.draft_round, p.draft_overall_pick,
            c.combine_height, c.combine_weight, c.hand_size, c.arm_length, c.wing_span,
            c.ten_yd_split, c.forty, c.vertical, c.broad_jump, c.three_cone, c.short_shuttle, c.bench_reps,
            c.ngs_athleticism_score, c.ngs_college_production_score, c.ngs_final_score
        FROM players p
        JOIN combine_results c ON p.nfl_id = c.nfl_id
        WHERE p.nfl_position IN ('WR', 'TE', 'RB', 'FB')
        ORDER BY p.draft_year, p.display_name
    """).df()

    # Derived Combine Physical Indices
    players_df["is_wr"] = (players_df["nfl_position"] == "WR").astype(float)
    players_df["is_te"] = (players_df["nfl_position"] == "TE").astype(float)
    players_df["bmi"] = (players_df["combine_weight"] * 703.0) / (players_df["combine_height"] ** 2)
    players_df["speed_score"] = (players_df["combine_weight"] * 200.0) / (players_df["forty"] ** 4)
    players_df["burst_score"] = players_df["vertical"] + players_df["broad_jump"]
    players_df["agility_score"] = players_df["three_cone"] + players_df["short_shuttle"]
    players_df["catch_radius"] = (
        players_df["arm_length"] + players_df["wing_span"] + players_df["hand_size"]
    )

    # 2. Combine Movement Tracking Kinematics & Temporal Trajectory Profiles
    tracking_df = con.execute("""
        SELECT 
            ct.nfl_id, ct.drill_type, ct.drill_name, ct.attempt, ct.time,
            ct.x, ct.y, ct.s, ct.a, ct.dis, ct.dir
        FROM combine_tracking ct
        JOIN players p ON ct.nfl_id = p.nfl_id
        WHERE p.nfl_position IN ('WR', 'TE', 'RB', 'FB')
          AND ct.entity_type = 'PLAYER'
        ORDER BY ct.nfl_id, ct.drill_name, ct.attempt, ct.time
    """).df()

    tracking_df["prev_dir"] = tracking_df.groupby(["nfl_id", "drill_name", "attempt"])["dir"].shift(1)
    tracking_df["prev_s"] = tracking_df.groupby(["nfl_id", "drill_name", "attempt"])["s"].shift(1)
    diff = (tracking_df["dir"] - tracking_df["prev_dir"] + 180.0) % 360.0 - 180.0
    tracking_df["delta_dir"] = diff.abs()
    tracking_df["tangential_a"] = (tracking_df["s"] - tracking_df["prev_s"]) / 0.1
    tracking_df["centripetal_a"] = tracking_df["s"] * np.radians(tracking_df["delta_dir"]) / 0.1
    tracking_df["cod_efficiency"] = np.where(
        (tracking_df["delta_dir"] > 20.0) & (tracking_df["s"] > 2.0), tracking_df["s"], np.nan
    )

    # Temporal phase progression within each active drill (0-25%, 25-50%, 50-75%, 75-100%)
    active_tr = tracking_df[tracking_df["s"] > 1.5].copy()
    active_tr["frame_idx"] = active_tr.groupby(["nfl_id", "drill_name", "attempt"]).cumcount()
    active_tr["frame_cnt"] = active_tr.groupby(["nfl_id", "drill_name", "attempt"])["s"].transform("count")
    active_tr["rel_pos"] = active_tr["frame_idx"] / np.maximum(1, active_tr["frame_cnt"] - 1)

    # A. 40-Yard Dash Tracking Kinematics & Temporal Phase Curve
    forty_tr = tracking_df[tracking_df["drill_type"] == "FORTY_YARD_DASH"]
    forty_feats = forty_tr.groupby("nfl_id").agg(
        tr40_max_s=("s", "max"),
        tr40_p90_s=("s", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) > 0 else np.nan),
        tr40_mean_s=("s", "mean"),
        tr40_max_a=("a", "max"),
        tr40_p90_a=("a", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) > 0 else np.nan),
        tr40_max_tang_a=("tangential_a", lambda x: np.percentile(x.dropna(), 95) if len(x.dropna()) > 0 else np.nan),
    ).reset_index()

    # 40-Yd Dash Temporal Phase Encodings (Early Burst 0-25%, Drive 25-50%, Top-End 50-75%, Finish 75-100%)
    forty_act = active_tr[active_tr["drill_type"] == "FORTY_YARD_DASH"]
    if len(forty_act) > 0:
        p1 = forty_act[forty_act["rel_pos"] < 0.25].groupby("nfl_id")["a"].max().rename("tr40_phase1_burst_a")
        p2 = forty_act[(forty_act["rel_pos"] >= 0.25) & (forty_act["rel_pos"] < 0.50)].groupby("nfl_id")["s"].mean().rename("tr40_phase2_drive_s")
        p4 = forty_act[forty_act["rel_pos"] >= 0.75].groupby("nfl_id")["s"].mean().rename("tr40_phase4_finish_s")
        forty_feats = forty_feats.merge(p1, on="nfl_id", how="left").merge(p2, on="nfl_id", how="left").merge(p4, on="nfl_id", how="left")
        forty_feats["tr40_speed_retention"] = forty_feats["tr40_phase4_finish_s"] / forty_feats["tr40_max_s"]

    # B. Agility Drill Tracking Features (3-Cone + Short Shuttle)
    agil_tr = tracking_df[tracking_df["drill_type"].isin(["THREE_CONE_DRILL", "SHORT_SHUTTLE"])]
    agil_feats = agil_tr.groupby("nfl_id").agg(
        tragil_max_s=("s", "max"),
        tragil_mean_s=("s", "mean"),
        tragil_max_a=("a", "max"),
        tragil_p90_a=("a", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) > 0 else np.nan),
        tragil_max_centrip=("centripetal_a", lambda x: np.percentile(x.dropna(), 95) if len(x.dropna()) > 0 else np.nan),
        tragil_cod_speed=("cod_efficiency", "mean"),
    ).reset_index()

    # C. Positional Route / Skill Drill Tracking Features (SKILL_DRILLS_WR / SKILL_DRILLS_TE)
    route_tr = tracking_df[tracking_df["drill_type"].isin(["SKILL_DRILLS_WR", "SKILL_DRILLS_TE"])]
    route_feats = route_tr.groupby("nfl_id").agg(
        trroute_drills_count=("drill_name", "nunique"),
        trroute_max_s=("s", "max"),
        trroute_p90_s=("s", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) > 0 else np.nan),
        trroute_mean_s=("s", "mean"),
        trroute_max_a=("a", "max"),
        trroute_p90_a=("a", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) > 0 else np.nan),
        trroute_max_decel=("tangential_a", lambda x: np.percentile(x.dropna(), 5) if len(x.dropna()) > 0 else np.nan),
        trroute_max_centrip=("centripetal_a", lambda x: np.percentile(x.dropna(), 95) if len(x.dropna()) > 0 else np.nan),
        trroute_cod_speed=("cod_efficiency", "mean"),
    ).reset_index()

    specific_drills = [
        "GAUNTLET_DRILL",
        "GO_ROUTE_RIGHT",
        "SLANT_ROUTE_LEFT",
        "DAGGER_ROUTE_LEFT",
        "CURL_ROUTE_RIGHT",
        "OVER_SHOULDER_ADJUST",
    ]
    for dname in specific_drills:
        prefix = f"drill_{dname.lower()[:8]}"
        sub = (
            route_tr[route_tr["drill_name"] == dname]
            .groupby("nfl_id")
            .agg(
                **{
                    f"{prefix}_max_s": ("s", "max"),
                    f"{prefix}_max_a": ("a", "max"),
                    f"{prefix}_cod_s": ("cod_efficiency", "mean"),
                }
            )
            .reset_index()
        )
        route_feats = route_feats.merge(sub, on="nfl_id", how="left")

    # Late-drill speed retention in Gauntlet & Route drills
    route_act = active_tr[active_tr["drill_type"].isin(["SKILL_DRILLS_WR", "SKILL_DRILLS_TE"])]
    if len(route_act) > 0:
        late_r = route_act[route_act["rel_pos"] >= 0.75].groupby(["nfl_id", "drill_name", "attempt"])["s"].mean().reset_index().rename(columns={"s": "late_s"})
        peak_r = route_act.groupby(["nfl_id", "drill_name", "attempt"])["s"].max().reset_index().rename(columns={"s": "peak_s"})
        ret_df = peak_r.merge(late_r, on=["nfl_id", "drill_name", "attempt"])
        ret_df["ret"] = ret_df["late_s"] / ret_df["peak_s"]
        g_ret = ret_df[ret_df["drill_name"] == "GAUNTLET_DRILL"].groupby("nfl_id")["ret"].mean().rename("gauntlet_speed_retention")
        r_ret = ret_df.groupby("nfl_id")["ret"].mean().rename("route_late_speed_retention")
        route_feats = route_feats.merge(g_ret, on="nfl_id", how="left").merge(r_ret, on="nfl_id", how="left")

    df = (
        players_df.merge(forty_feats, on="nfl_id", how="left")
        .merge(agil_feats, on="nfl_id", how="left")
        .merge(route_feats, on="nfl_id", how="left")
    )
    df["route_to_40_speed_ratio"] = df["trroute_max_s"] / df["tr40_max_s"]

    # 3. Play-by-Play Targets: Rookie Season, All Seasons, and Late-Game / Q4 Clutch
    plays = con.execute("""
        SELECT 
            pp.game_id,
            pp.play_id,
            pp.nfl_id,
            pp.team_abbr,
            p.display_name,
            p.nfl_position,
            p.draft_year,
            g.season,
            g.week,
            g.home_team_abbr,
            g.visitor_team_abbr,
            pp.quarter,
            pp.down,
            pp.yards_to_go,
            pp.pre_snap_home_score,
            pp.pre_snap_visitor_score,
            pp.pre_snap_home_team_win_probability,
            pp.pre_snap_visitor_team_win_probability,
            pp.home_team_win_probability_added,
            pp.visitor_team_win_probility_added,
            pp.expected_points_added,
            pp.rec_yards,
            pp.rush_yards,
            pp.yards_after_catch,
            pp.expected_yards_after_catch,
            pp.target,
            pp.route_ran,
            pp.separation_at_pass_forward,
            pp.cushion,
            pp.play_nullified_by_penalty,
            pp.fumble_lost,
            pp.play_description
        FROM player_play pp
        JOIN players p ON pp.nfl_id = p.nfl_id
        JOIN games g ON pp.game_id = g.game_id
        WHERE p.nfl_position IN ('WR', 'TE', 'RB', 'FB')
          AND (pp.play_nullified_by_penalty IS NULL OR pp.play_nullified_by_penalty = 'N')
    """).df()

    if close_con:
        con.close()

    plays["is_td"] = plays.apply(_is_player_td, axis=1)
    plays["rec_yards_clean"] = plays["rec_yards"].fillna(0.0)
    plays["rush_yards_clean"] = plays["rush_yards"].fillna(0.0)
    plays["scrimmage_yards"] = plays["rec_yards_clean"] + plays["rush_yards_clean"]
    plays["yac_clean"] = plays["yards_after_catch"].fillna(0.0)
    plays["is_route"] = plays["route_ran"].notna().astype(int)
    plays["is_target"] = plays["target"].fillna(False).astype(int)
    plays["is_catch"] = plays["rec_yards"].notna().astype(int)
    plays["is_explosive"] = (plays["scrimmage_yards"] >= 15).astype(int)

    is_home = plays["team_abbr"] == plays["home_team_abbr"]
    plays["opponent_abbr"] = np.where(is_home, plays["visitor_team_abbr"], plays["home_team_abbr"])
    plays["is_home"] = is_home.astype(int)
    plays["team_wp"] = np.where(
        is_home, plays["pre_snap_home_team_win_probability"], plays["pre_snap_visitor_team_win_probability"]
    )
    plays["team_wpa"] = np.where(
        is_home, plays["home_team_win_probability_added"], plays["visitor_team_win_probility_added"]
    ).astype(float)
    plays["team_wpa"] = np.nan_to_num(plays["team_wpa"], nan=0.0)
    plays["target_wpa"] = np.where(
        (plays["is_catch"] == 1) | (plays["rush_yards_clean"] > 0),
        np.maximum(0.0, plays["team_wpa"]),
        0.0,
    )
    plays["score_diff"] = np.abs(plays["pre_snap_home_score"] - plays["pre_snap_visitor_score"])

    plays["is_q4"] = (plays["quarter"] >= 4).astype(int)
    plays["is_clutch"] = (
        (plays["quarter"] >= 4)
        & ((plays["score_diff"] <= 8) | ((plays["team_wp"] >= 0.15) & (plays["team_wp"] <= 0.85)))
    ).astype(int)
    plays["is_rookie_play"] = (plays["season"] == plays["draft_year"]).astype(int)

    # Game-level table (All seasons + flag for rookie season)
    plays["q4_yards_play"] = np.where(plays["is_q4"] == 1, plays["scrimmage_yards"], 0.0)
    plays["q4_td_play"] = np.where(plays["is_q4"] == 1, plays["is_td"], 0)
    plays["q4_route_play"] = np.where(plays["is_q4"] == 1, plays["is_route"], 0)
    plays["clutch_yards_play"] = np.where(plays["is_clutch"] == 1, plays["scrimmage_yards"], 0.0)
    plays["clutch_td_play"] = np.where(plays["is_clutch"] == 1, plays["is_td"], 0)
    plays["clutch_route_play"] = np.where(plays["is_clutch"] == 1, plays["is_route"], 0)
    plays["clutch_wpa_play"] = np.where(plays["is_clutch"] == 1, plays["target_wpa"], 0.0)
    plays["q4_explosive_play"] = np.where(plays["is_q4"] == 1, plays["is_explosive"], 0)

    player_games_df = (
        plays.groupby(
            [
                "nfl_id",
                "display_name",
                "nfl_position",
                "draft_year",
                "season",
                "week",
                "game_id",
                "team_abbr",
                "opponent_abbr",
                "is_home",
                "is_rookie_play",
            ]
        )
        .agg(
            snaps=("play_id", "count"),
            routes=("is_route", "sum"),
            targets=("is_target", "sum"),
            catches=("is_catch", "sum"),
            rec_yards=("rec_yards_clean", "sum"),
            rush_yards=("rush_yards_clean", "sum"),
            total_yards=("scrimmage_yards", "sum"),
            tds=("is_td", "sum"),
            yac=("yac_clean", "sum"),
            q4_routes=("q4_route_play", "sum"),
            q4_yards=("q4_yards_play", "sum"),
            q4_tds=("q4_td_play", "sum"),
            q4_explosive=("q4_explosive_play", "sum"),
            clutch_routes=("clutch_route_play", "sum"),
            clutch_yards=("clutch_yards_play", "sum"),
            clutch_tds=("clutch_td_play", "sum"),
            clutch_wpa=("clutch_wpa_play", "sum"),
            avg_sep=("separation_at_pass_forward", "mean"),
            avg_cushion=("cushion", "mean"),
            epa=("expected_points_added", "sum"),
        )
        .reset_index()
    )

    # Rookie Season Aggregation (season == draft_year)
    rk_games = player_games_df[player_games_df["is_rookie_play"] == 1]
    rk_summary = (
        rk_games.groupby("nfl_id")
        .agg(
            games_played=("game_id", "nunique"),
            total_snaps=("snaps", "sum"),
            total_routes=("routes", "sum"),
            total_targets=("targets", "sum"),
            total_catches=("catches", "sum"),
            total_yards=("total_yards", "sum"),
            total_tds=("tds", "sum"),
            yards_per_game=("total_yards", "mean"),
            tds_per_game=("tds", "mean"),
            routes_per_game=("routes", "mean"),
            targets_per_game=("targets", "mean"),
            rk_q4_yards=("q4_yards", "sum"),
            rk_q4_tds=("q4_tds", "sum"),
            rk_q4_explosive=("q4_explosive", "sum"),
            rk_clutch_yards=("clutch_yards", "sum"),
            rk_clutch_tds=("clutch_tds", "sum"),
            rk_clutch_wpa=("clutch_wpa", "sum"),
            mean_separation=("avg_sep", "mean"),
            mean_cushion=("avg_cushion", "mean"),
        )
        .reset_index()
    )

    # All-Seasons Career Late-Game & Clutch Aggregation (larger sample size for Q4 / Clutch stability)
    career_summary = (
        player_games_df.groupby("nfl_id")
        .agg(
            career_games=("game_id", "nunique"),
            career_yards_per_game=("total_yards", "mean"),
            career_tds_per_game=("tds", "mean"),
            career_q4_yards=("q4_yards", "sum"),
            career_q4_tds=("q4_tds", "sum"),
            career_q4_routes=("q4_routes", "sum"),
            career_q4_explosive=("q4_explosive", "sum"),
            career_clutch_yards=("clutch_yards", "sum"),
            career_clutch_tds=("clutch_tds", "sum"),
            career_clutch_routes=("clutch_routes", "sum"),
            career_clutch_wpa=("clutch_wpa", "sum"),
            career_total_routes=("routes", "sum"),
            career_total_yards=("total_yards", "sum"),
        )
        .reset_index()
    )

    # Compute Q4 vs Q1-Q3 Yards/Route Step-Up delta
    q13_routes = career_summary["career_total_routes"] - career_summary["career_q4_routes"]
    q13_yards = career_summary["career_total_yards"] - career_summary["career_q4_yards"]
    career_summary["q13_ypr"] = np.where(q13_routes >= 15, q13_yards / q13_routes, np.nan)
    career_summary["q4_ypr"] = np.where(
        career_summary["career_q4_routes"] >= 10,
        career_summary["career_q4_yards"] / career_summary["career_q4_routes"],
        np.nan,
    )
    career_summary["q4_vs_q13_ypr_diff"] = career_summary["q4_ypr"] - career_summary["q13_ypr"]
    # Normalize late-game totals per 17-game season played (so healthy durability + clutch rate are both reflected)
    seasons_equiv = np.maximum(1.0, career_summary["career_games"] / 15.0)
    career_summary["clutch_yards_per_game"] = career_summary["career_clutch_yards"] / seasons_equiv
    career_summary["q4_tds_per_game"] = career_summary["career_q4_tds"] / seasons_equiv
    career_summary["q4_explosive_per_game"] = career_summary["career_q4_explosive"] / seasons_equiv
    career_summary["clutch_wpa_per_season"] = career_summary["career_clutch_wpa"] / seasons_equiv

    df = df.merge(rk_summary, on="nfl_id", how="left").merge(career_summary, on="nfl_id", how="left")

    zero_fill_cols = [
        "games_played",
        "total_snaps",
        "total_routes",
        "total_targets",
        "total_catches",
        "total_yards",
        "total_tds",
        "yards_per_game",
        "tds_per_game",
        "routes_per_game",
        "targets_per_game",
        "rk_q4_yards",
        "rk_q4_tds",
        "rk_q4_explosive",
        "rk_clutch_yards",
        "rk_clutch_tds",
        "rk_clutch_wpa",
        "career_games",
        "career_yards_per_game",
        "career_tds_per_game",
        "career_q4_yards",
        "career_q4_tds",
        "career_q4_routes",
        "career_q4_explosive",
        "career_clutch_yards",
        "career_clutch_tds",
        "career_clutch_routes",
        "career_clutch_wpa",
        "clutch_yards_per_game",
        "q4_tds_per_game",
        "q4_explosive_per_game",
        "clutch_wpa_per_season",
    ]
    for col in zero_fill_cols:
        df[col] = df[col].fillna(0.0)

    df["yards_per_route"] = np.where(
        df["total_routes"] >= 20, df["total_yards"] / df["total_routes"], np.nan
    )
    df["q4_vs_q13_ypr_diff_filled"] = df["q4_vs_q13_ypr_diff"].fillna(0.0)

    return df, player_games_df, tracking_df


# ---------------------------------------------------------------------------
# Feature Tiers for the Neural Network
# ---------------------------------------------------------------------------
PHYSICAL_FEATURES = [
    "is_wr",
    "is_te",
    "combine_height",
    "combine_weight",
    "bmi",
    "hand_size",
    "arm_length",
    "wing_span",
    "catch_radius",
    "ten_yd_split",
    "forty",
    "speed_score",
    "vertical",
    "broad_jump",
    "burst_score",
    "three_cone",
    "short_shuttle",
    "agility_score",
    "bench_reps",
]

MOVEMENT_FEATURES = [
    "tr40_max_s",
    "tr40_p90_s",
    "tr40_mean_s",
    "tr40_max_a",
    "tr40_p90_a",
    "tr40_max_tang_a",
    "tr40_phase1_burst_a",
    "tr40_phase2_drive_s",
    "tr40_phase4_finish_s",
    "tr40_speed_retention",
    "tragil_max_s",
    "tragil_mean_s",
    "tragil_max_a",
    "tragil_p90_a",
    "tragil_max_centrip",
    "tragil_cod_speed",
    "trroute_max_s",
    "trroute_p90_s",
    "trroute_mean_s",
    "trroute_max_a",
    "trroute_p90_a",
    "trroute_max_decel",
    "trroute_max_centrip",
    "trroute_cod_speed",
    "route_to_40_speed_ratio",
    "gauntlet_speed_retention",
    "route_late_speed_retention",
    "drill_gauntlet_max_s",
    "drill_gauntlet_max_a",
    "drill_go_route_max_s",
    "drill_go_route_max_a",
    "drill_slant_ro_max_s",
    "drill_slant_ro_max_a",
    "drill_dagger_r_max_s",
    "drill_dagger_r_max_a",
    "drill_curl_rou_max_s",
    "drill_curl_rou_max_a",
    "drill_over_sho_max_s",
    "drill_over_sho_max_a",
]

NGS_PRE_DRAFT_FEATURES = [
    "ngs_athleticism_score",
    "ngs_college_production_score",
    "ngs_final_score",
]

DRAFT_CAPITAL_FEATURES = [
    "draft_overall_pick",
    "draft_round",
]

FEATURE_TIERS = {
    "Tier 1: Pure Combine Physical + Movement Tracking": PHYSICAL_FEATURES + MOVEMENT_FEATURES,
    "Tier 2: Combine + Movement + NGS Pre-Draft Scores": PHYSICAL_FEATURES + MOVEMENT_FEATURES + NGS_PRE_DRAFT_FEATURES,
    "Tier 3: Combine + Movement + NGS + Draft Capital": PHYSICAL_FEATURES + MOVEMENT_FEATURES + NGS_PRE_DRAFT_FEATURES + DRAFT_CAPITAL_FEATURES,
}

TARGET_COLUMNS = [
    "yards_per_game",
    "tds_per_game",
    "clutch_yards_per_game",
    "q4_tds_per_game",
    "q4_explosive_per_game",
    "clutch_wpa_per_season",
]

TARGET_LABELS = {
    "yards_per_game": "Rookie Yards / Game",
    "tds_per_game": "Rookie TDs / Game",
    "clutch_yards_per_game": "Q4 Clutch Yards / Season",
    "q4_tds_per_game": "4th-Quarter TDs / Season",
    "q4_explosive_per_game": "Q4 Explosive Plays (15+ yd) / Season",
    "clutch_wpa_per_season": "Q4 Clutch Win Prob Added (WPA / Season)",
}


# ---------------------------------------------------------------------------
# Dual-Branch Multi-Task Neural Network (Pure NumPy with Adam + Residual Skip)
# ---------------------------------------------------------------------------
class DualBranchCombineNN:
    """
    Dual-Branch Multi-Task Neural Network with Residual Skip Connection:
      - Branch A (Physical & Pre-Draft Encoder): Dense(d_phys -> h_phys) + LayerNorm + SiLU/GELU
      - Branch B (Combine Movement Kinematics Encoder): Dense(d_move -> h_move) + LayerNorm + SiLU/GELU
      - Fusion Layer: Concat([Branch A, Branch B]) -> Dense(h_phys + h_move -> h_joint) + SiLU + Dropout
      - Multi-Task Output Head + Residual Linear Skip: Predicts 6 targets simultaneously
        using Smooth-L1 (Huber) loss and Adam optimizer with Cosine Annealing.
    """

    def __init__(self, phys_indices, move_indices, n_targets, h_phys=16, h_move=20, h_joint=24, lr=0.015, weight_decay=0.35, epochs=320, seed=42):
        self.phys_idx = np.asarray(phys_indices, dtype=int)
        self.move_idx = np.asarray(move_indices, dtype=int)
        self.n_targets = n_targets
        self.h_phys = h_phys
        self.h_move = h_move
        self.h_joint = h_joint
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed

        # Preprocessing parameters learned on train fold
        self.col_medians = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.feature_gate = None

        # Network weights
        self.params = {}

    @staticmethod
    def _silu(z):
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
        return z * sig, sig * (1.0 + z * (1.0 - sig))

    def _preprocess_fit(self, X, Y):
        X = np.asarray(X, dtype=float).copy()
        Y = np.asarray(Y, dtype=float).copy()
        n, d = X.shape

        # Compute non-missing coverage and in-fold correlation gate (strictly on training fold)
        coverage = np.mean(np.isfinite(X), axis=0)
        self.col_medians = np.zeros(d)
        for j in range(d):
            finite_vals = X[np.isfinite(X[:, j]), j]
            self.col_medians[j] = float(np.median(finite_vals)) if len(finite_vals) > 0 else 0.0
            X[~np.isfinite(X[:, j]), j] = self.col_medians[j]

        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std[self.x_std < 1e-6] = 1.0
        Xs = np.clip((X - self.x_mean) / self.x_std, -3.5, 3.5)

        self.y_mean = Y.mean(axis=0)
        self.y_std = Y.std(axis=0)
        self.y_std[self.y_std < 1e-6] = 1.0
        Ys = (Y - self.y_mean) / self.y_std

        # Target-specific in-fold correlation & coverage attention gate (d, n_targets)
        # Features with higher training-fold correlation and coverage get stronger skip/input weight
        corr_mat = np.abs((Xs.T @ Ys) / n)  # (d, n_targets)
        cov_weight = np.sqrt(np.clip(coverage, 0.15, 1.0))[:, None]
        raw_gate = (corr_mat ** 1.25) * cov_weight
        # Keep top-10 features per target active in the skip gate to prevent overfitting on N=150
        for t in range(self.n_targets):
            thresh = np.sort(raw_gate[:, t])[-min(10, d)]
            raw_gate[raw_gate[:, t] < thresh, t] *= 0.10
        gate_max = np.maximum(1e-4, np.max(raw_gate, axis=0, keepdims=True))
        self.feature_gate = raw_gate / gate_max

        # Scale Xs by overall multi-task relevance so the hidden branches focus on high-signal inputs
        input_relevance = np.clip(np.max(corr_mat, axis=1) * np.sqrt(np.clip(coverage, 0.2, 1.0)) * 2.5, 0.30, 1.4)
        self.input_scale = input_relevance
        Xs = Xs * self.input_scale[None, :]
        return Xs, Ys

    def _preprocess_transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            mask = ~np.isfinite(X[:, j])
            X[mask, j] = self.col_medians[j]
        Xs = np.clip((X - self.x_mean) / self.x_std, -3.5, 3.5)
        return Xs * self.input_scale[None, :]

    def fit(self, X, Y):
        Xs, Ys = self._preprocess_fit(X, Y)
        n, d = Xs.shape
        d_p = len(self.phys_idx)
        d_m = len(self.move_idx)
        rng = np.random.default_rng(self.seed)

        # Initialize Dual-Branch + Residual Skip parameters
        self.params = {
            "Wp": rng.normal(0.0, np.sqrt(1.0 / max(1, d_p)), size=(d_p, self.h_phys)),
            "bp": np.zeros(self.h_phys),
            "Wm": rng.normal(0.0, np.sqrt(1.0 / max(1, d_m)), size=(d_m, self.h_move)),
            "bm": np.zeros(self.h_move),
            "Wj": rng.normal(0.0, np.sqrt(1.0 / (self.h_phys + self.h_move)), size=(self.h_phys + self.h_move, self.h_joint)),
            "bj": np.zeros(self.h_joint),
            "Wo": rng.normal(0.0, 0.15 / np.sqrt(self.h_joint), size=(self.h_joint, self.n_targets)),
            "bo": np.zeros(self.n_targets),
            # Residual skip connection initialized from regularized Ridge solution for strong stability on N=150
            "W_skip": np.linalg.solve(Xs.T @ Xs + 45.0 * np.eye(d), Xs.T @ Ys),
        }

        # Adam state
        m_state = {k: np.zeros_like(v) for k, v in self.params.items()}
        v_state = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        delta = 1.25  # Huber loss threshold in standardized units

        Xp = Xs[:, self.phys_idx]
        Xm = Xs[:, self.move_idx]

        for epoch in range(1, self.epochs + 1):
            # Cosine annealing learning rate
            lr_t = self.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / self.epochs))

            # Forward pass
            zp = Xp @ self.params["Wp"] + self.params["bp"]
            ap, dap = self._silu(zp)

            zm = Xm @ self.params["Wm"] + self.params["bm"]
            am, dam = self._silu(zm)

            concat_pm = np.concatenate([ap, am], axis=1)
            zj = concat_pm @ self.params["Wj"] + self.params["bj"]
            aj, daj = self._silu(zj)

            # Nonlinear residual correction + skip path
            y_pred = aj @ self.params["Wo"] + self.params["bo"] + Xs @ self.params["W_skip"]

            # Smooth-L1 / Huber loss gradient
            err = y_pred - Ys
            grad_out = np.where(np.abs(err) <= delta, err, delta * np.sign(err)) / n

            # Backpropagation
            grads = {}
            grads["Wo"] = aj.T @ grad_out + self.weight_decay * self.params["Wo"]
            grads["bo"] = grad_out.sum(axis=0)
            grads["W_skip"] = (Xs.T @ grad_out) + (self.weight_decay * 1.5) * self.params["W_skip"]

            d_aj = grad_out @ self.params["Wo"].T
            d_zj = d_aj * daj
            grads["Wj"] = concat_pm.T @ d_zj + self.weight_decay * self.params["Wj"]
            grads["bj"] = d_zj.sum(axis=0)

            d_concat = d_zj @ self.params["Wj"].T
            d_ap = d_concat[:, : self.h_phys]
            d_am = d_concat[:, self.h_phys :]

            d_zp = d_ap * dap
            grads["Wp"] = Xp.T @ d_zp + self.weight_decay * self.params["Wp"]
            grads["bp"] = d_zp.sum(axis=0)

            d_zm = d_am * dam
            grads["Wm"] = Xm.T @ d_zm + self.weight_decay * self.params["Wm"]
            grads["bm"] = d_zm.sum(axis=0)

            # Adam parameter update
            for k in self.params:
                m_state[k] = beta1 * m_state[k] + (1.0 - beta1) * grads[k]
                v_state[k] = beta2 * v_state[k] + (1.0 - beta2) * (grads[k] ** 2)
                m_hat = m_state[k] / (1.0 - beta1 ** epoch)
                v_hat = v_state[k] / (1.0 - beta2 ** epoch)
                self.params[k] -= lr_t * m_hat / (np.sqrt(v_hat) + eps)

        return self

    def predict(self, X):
        Xs = self._preprocess_transform(X)
        Xp = Xs[:, self.phys_idx]
        Xm = Xs[:, self.move_idx]

        ap, _ = self._silu(Xp @ self.params["Wp"] + self.params["bp"])
        am, _ = self._silu(Xm @ self.params["Wm"] + self.params["bm"])
        concat_pm = np.concatenate([ap, am], axis=1)
        aj, _ = self._silu(concat_pm @ self.params["Wj"] + self.params["bj"])

        y_pred_std = aj @ self.params["Wo"] + self.params["bo"] + Xs @ self.params["W_skip"]
        y_pred = y_pred_std * self.y_std + self.y_mean

        # Non-negative clipping for count/yardage rate targets (indices 0..4); index 5 is delta YPRR (can be negative)
        y_pred[:, :5] = np.maximum(0.0, y_pred[:, :5])
        return y_pred


# ---------------------------------------------------------------------------
# 5-Fold Cross-Validation Ensemble & Feature Importance Engine
# ---------------------------------------------------------------------------
def train_and_evaluate_tiers(df, k=5, seed=42):
    """
    Trains a 5-Fold Cross-Validated ensemble of DualBranchCombineNN models for each
    Feature Tier, returning out-of-sample predictions, fold uncertainty (std),
    evaluation metrics (Pearson r, p-val, Spearman rho, MAE), and feature importances.
    """
    results = {}
    Y = df[TARGET_COLUMNS].to_numpy(dtype=float)
    n = len(df)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    for tier_name, feat_cols in FEATURE_TIERS.items():
        X_df = df[feat_cols]
        phys_idx = [i for i, c in enumerate(feat_cols) if c not in MOVEMENT_FEATURES]
        move_idx = [i for i, c in enumerate(feat_cols) if c in MOVEMENT_FEATURES]

        oof_preds = np.zeros_like(Y)
        models = []

        for f_i, val_idx in enumerate(folds):
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False

            model = DualBranchCombineNN(
                phys_indices=phys_idx,
                move_indices=move_idx,
                n_targets=len(TARGET_COLUMNS),
                seed=seed + f_i * 17,
            )
            model.fit(X_df.iloc[train_mask].to_numpy(dtype=float, na_value=np.nan), Y[train_mask])
            oof_preds[val_idx] = model.predict(X_df.iloc[val_idx].to_numpy(dtype=float, na_value=np.nan))
            models.append(model)

        # Ensemble prediction & uncertainty across all 5 fold models
        all_fold_preds = np.stack(
            [m.predict(X_df.to_numpy(dtype=float, na_value=np.nan)) for m in models], axis=0
        )
        ensemble_std = all_fold_preds.std(axis=0)

        # Metrics per target
        metrics = []
        for t_i, t_col in enumerate(TARGET_COLUMNS):
            y_true = Y[:, t_i]
            y_hat = oof_preds[:, t_i]
            r, p = pearson_r_pval(y_true, y_hat)
            rho, prho = spearman_rho_pval(y_true, y_hat)
            mae = float(np.mean(np.abs(y_true - y_hat)))
            rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
            metrics.append(
                {
                    "target": t_col,
                    "target_label": TARGET_LABELS[t_col],
                    "r": r,
                    "p_val": p,
                    "spearman_rho": rho,
                    "mae": mae,
                    "rmse": rmse,
                }
            )

        # Feature sensitivity / effective importance across the 5-fold ensemble
        importances = compute_feature_importances(models, X_df, feat_cols)

        results[tier_name] = {
            "feature_cols": feat_cols,
            "models": models,
            "oof_preds": oof_preds,
            "ensemble_std": ensemble_std,
            "metrics_df": pd.DataFrame(metrics),
            "importances_df": importances,
        }

    return results


def compute_feature_importances(models, X_df, feat_cols):
    """
    Computes standardized perturbation sensitivity (directional & magnitude importance)
    of each Combine/Movement feature on each target across the 5-fold NN ensemble.
    """
    X_raw = X_df.to_numpy(dtype=float, na_value=np.nan)
    base_preds = np.mean([m.predict(X_raw) for m in models], axis=0)
    rows = []

    for j, col in enumerate(feat_cols):
        col_vals = X_raw[:, j]
        sd = np.nanstd(col_vals)
        if not np.isfinite(sd) or sd < 1e-6:
            continue
        X_plus = X_raw.copy()
        X_Minus = X_raw.copy()
        X_plus[:, j] = col_vals + 0.5 * sd
        X_Minus[:, j] = col_vals - 0.5 * sd

        pred_plus = np.mean([m.predict(X_plus) for m in models], axis=0)
        pred_minus = np.mean([m.predict(X_Minus) for m in models], axis=0)
        delta = np.mean(pred_plus - pred_minus, axis=0)

        cat = (
            "Movement Tracking"
            if col in MOVEMENT_FEATURES
            else (
                "NGS Pre-Draft Score"
                if col in NGS_PRE_DRAFT_FEATURES
                else ("Draft Capital" if col in DRAFT_CAPITAL_FEATURES else "Combine Physical")
            )
        )
        row = {"feature": col, "category": cat}
        for t_i, t_col in enumerate(TARGET_COLUMNS):
            row[f"effect_{t_col}"] = float(delta[t_i])
            row[f"abs_effect_{t_col}"] = float(abs(delta[t_i]))
        rows.append(row)

    return pd.DataFrame(rows)


def predict_custom_prospect(models, feat_cols, custom_feature_dict):
    """
    Runs a single prospect's feature dictionary through the 5-fold NN ensemble,
    returning mean predictions and fold confidence intervals (std).
    """
    x_vec = np.array([[custom_feature_dict.get(c, np.nan) for c in feat_cols]], dtype=float)
    fold_preds = np.stack([m.predict(x_vec)[0] for m in models], axis=0)
    mean_pred = fold_preds.mean(axis=0)
    std_pred = fold_preds.std(axis=0)
    return {
        t_col: {"mean": float(mean_pred[i]), "std": float(std_pred[i])}
        for i, t_col in enumerate(TARGET_COLUMNS)
    }


def project_player_game(player_row, oof_pred_dict, routes_in_game=20, game_situation="Standard Full Game (Q1-Q4)", is_clutch_close=False):
    """
    Conditions a player's Neural Network Combine projection on specific game opportunity
    (routes_in_game) and late-game situation (Standard vs Q4/OT vs Q4 One-Score Clutch).
    """
    base_ypg = max(2.0, float(oof_pred_dict["yards_per_game"]))
    base_tdpg = max(0.01, float(oof_pred_dict["tds_per_game"]))
    clutch_ypg = max(1.0, float(oof_pred_dict["clutch_yards_per_game"]))
    q4_tdpg = max(0.005, float(oof_pred_dict["q4_tds_per_game"]))
    q4_exp_pg = max(0.01, float(oof_pred_dict["q4_explosive_per_game"]))
    clutch_wpa_season = float(oof_pred_dict.get("clutch_wpa_per_season", 0.05))
    delta_ypr = float(player_row.get("q4_vs_q13_ypr_diff_filled", 0.0)) if player_row is not None else 0.0

    # Baseline routes per game across rookie dataset is ~15.1 routes/game (or player's own avg if > 8)
    ref_routes = 16.5
    opp_multiplier = routes_in_game / ref_routes

    # Late-game situational multiplier derived from player's Q4 step-up & clutch profile
    if game_situation == "4th Quarter / Overtime Only":
        # Projecting a 4th-quarter stint (e.g. 6-12 routes in Q4)
        q4_step_mult = np.clip(1.0 + (delta_ypr / 1.75), 0.75, 1.55)
        proj_yards = (base_ypg / ref_routes) * routes_in_game * q4_step_mult
        proj_tds = ((q4_tdpg / 16.0) / 4.2) * routes_in_game
        explosive_prob = 1.0 - math.exp(-max(0.02, ((q4_exp_pg / 16.0) / 4.2) * routes_in_game))
        expected_wpa = proj_tds * 0.18 + (proj_yards / 100.0) * 0.09
    elif game_situation == "Clutch Q4 (One-Score Game / 2-Min Drill)":
        q4_step_mult = np.clip(1.05 + (delta_ypr / 1.5), 0.80, 1.70)
        proj_yards = (
            0.5 * (base_ypg / ref_routes) * routes_in_game * q4_step_mult
            + 0.5 * ((clutch_ypg / 16.0) / 3.5) * routes_in_game
        )
        proj_tds = ((q4_tdpg / 16.0) / 3.5) * routes_in_game * (1.20 if is_clutch_close else 1.0)
        explosive_prob = 1.0 - math.exp(-max(0.03, ((q4_exp_pg / 16.0) / 3.5) * routes_in_game * 1.15))
        expected_wpa = proj_tds * 0.28 + (proj_yards / 100.0) * 0.16
    else:
        # Full Game projection
        clutch_boost = 1.0 + (0.08 * np.clip(delta_ypr, -0.5, 1.0) if is_clutch_close else 0.0)
        proj_yards = base_ypg * opp_multiplier * clutch_boost
        proj_tds = base_tdpg * opp_multiplier * (1.12 if is_clutch_close else 1.0)
        explosive_prob = 1.0 - math.exp(-max(0.05, (proj_yards / 28.0)))
        expected_wpa = proj_tds * (0.22 if is_clutch_close else 0.14) + (proj_yards / 100.0) * 0.08

    prob_1plus_td = 1.0 - math.exp(-proj_tds)
    return {
        "projected_yards": float(proj_yards),
        "projected_tds": float(proj_tds),
        "prob_1plus_td": float(prob_1plus_td),
        "prob_explosive_15yd": float(explosive_prob),
        "expected_wpa": float(expected_wpa),
    }


if __name__ == "__main__":
    print("Building Skill Position Combine & Movement dataset from DuckDB...")
    df, player_games_df, _ = build_skill_position_dataset()
    print(f"Loaded {len(df)} skill-position players and {len(player_games_df)} player-games.")

    tier_results = train_and_evaluate_tiers(df, k=5, seed=42)
    for tier_name, res in tier_results.items():
        print("\n" + "=" * 90)
        print(f"5-FOLD OUT-OF-SAMPLE EVALUATION: {tier_name}")
        print("=" * 90)
        m_df = res["metrics_df"].copy()
        m_df["r"] = m_df["r"].round(3)
        m_df["p_val"] = m_df["p_val"].round(4)
        m_df["spearman_rho"] = m_df["spearman_rho"].round(3)
        m_df["mae"] = m_df["mae"].round(3)
        print(m_df[["target_label", "r", "p_val", "spearman_rho", "mae"]].to_string(index=False))
