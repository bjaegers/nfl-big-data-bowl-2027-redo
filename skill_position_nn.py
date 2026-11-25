"""
Skill Position Combine & Movement Modeling Engine
=================================================
Multi-Model Suite & Team-Impact Projector:
  Model Families:
    1. Dual-Branch Multi-Task Neural Network (with Ridge residual skip & Smooth-L1 loss)
    2. Bayesian Hierarchical Model (Position partial pooling, exposure precision weighting,
       ARD shrinkage, analytical 10th-90th percentile Floor/Ceiling credible intervals,
       and week-by-week Bayesian prior-to-posterior updating)
    3. XGBoost-style Gradient Boosted Decision Trees (with native missing-value split routing)
    4. Hybrid Stack Ensemble (combining NN + Bayesian Hierarchical + XGBoost/GBDT)

  Target Suites (11 Targets):
    A. Box-Score & Late-Game Clutch Suite (6 targets):
       - Rookie Yards / Game (`yards_per_game`)
       - Rookie TDs / Game (`tds_per_game`)
       - Q4 Clutch Yards / Season (`clutch_yards_per_game`)
       - 4th-Quarter TDs / Season (`q4_tds_per_game`)
       - Q4 Explosive Plays (15+ yd) / Season (`q4_explosive_per_game`)
       - Q4 Clutch Win Prob Added / Season (`clutch_wpa_per_season`)
    B. Beyond-Box-Score Team Impact & Offensive Gravity Suite (5 targets):
       - Chain-Moving 1st Downs / 100 Routes (`first_downs_per_100_routes`)
       - 3rd & 4th Down Money Conversions / Season (`third_fourth_conv_per_season`)
       - Expected Points Added on Targets / Season (`target_epa_per_season`)
       - Target-Earning Rate (`tprr_pct` = Targets / 100 Routes)
       - Offensive Gravity: Non-Target Team EPA / 100 Snaps (`gravity_non_target_epa_100`)
"""

import os
import math
import duckdb
import numpy as np
import pandas as pd

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_FULL_DB = os.path.join(_ROOT_DIR, "nfl_data.duckdb")
_LITE_DB = os.path.join(_ROOT_DIR, "nfl_data_lite.duckdb")
DB_PATH = _FULL_DB if os.path.exists(_FULL_DB) else _LITE_DB


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
# Feature Extraction & Team-Impact Pipeline from DuckDB
# ---------------------------------------------------------------------------
def build_skill_position_dataset(con=None):
    """
    Extracts and engineers:
      - Combine physical measurements & athletic indices from `combine_results`
      - Frame-by-frame Combine drill movement kinematics & phase profiles from `combine_tracking`
      - Box-score, 4th-Quarter Clutch, and Beyond-Box-Score Team Impact & Gravity targets
      - Individual player-game logs (`player_games_df`) across all rookie and career games
    """
    close_con = False
    if con is None:
        con = duckdb.connect(DB_PATH, read_only=True)
        close_con = True

    # 1. Base skill-position players + combine_results (strictly End-of-Combine information)
    players_df = con.execute("""
        SELECT 
            p.nfl_id, p.display_name, p.nfl_position, p.draft_year,
            p.college_name, p.college_conference,
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

    # 3. Play-by-Play Targets: Box Score, Q4 Clutch, and Beyond-Box-Score Team Impact & Gravity
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
            pp.in_motion_at_ball_snap,
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
    plays["yacoe_play"] = np.where(
        plays["rec_yards"].notna() & plays["expected_yards_after_catch"].notna(),
        plays["yards_after_catch"].fillna(0.0) - plays["expected_yards_after_catch"].fillna(0.0),
        0.0,
    )
    plays["is_route"] = plays["route_ran"].notna().astype(int)
    plays["is_target"] = plays["target"].fillna(False).astype(int)
    plays["is_catch"] = plays["rec_yards"].notna().astype(int)
    plays["is_explosive"] = (plays["scrimmage_yards"] >= 15).astype(int)
    plays["is_motion"] = plays["in_motion_at_ball_snap"].fillna(False).astype(int)

    # Chain-moving first downs & 3rd/4th down conversions earned by the player
    plays["is_first_down_earned"] = (
        (plays["scrimmage_yards"] > 0)
        & (plays["scrimmage_yards"] >= plays["yards_to_go"].fillna(99))
    ).astype(int)
    plays["is_money_down_conv"] = (
        plays["down"].isin([3, 4]) & (plays["is_first_down_earned"] == 1)
    ).astype(int)

    # EPA dimensions: Positive Target EPA Created vs. Non-Targeted Decoy Team EPA ("Offensive Gravity")
    plays["epa_clean"] = plays["expected_points_added"].fillna(0.0)
    plays["target_epa_play"] = np.where(
        (plays["is_target"] == 1) & (plays["epa_clean"] > 0), plays["epa_clean"], 0.0
    )
    plays["is_non_target_snap"] = (plays["is_target"] == 0).astype(int)
    plays["non_target_epa_play"] = np.where(
        (plays["is_target"] == 0) & (plays["epa_clean"] > 0), plays["epa_clean"], 0.0
    )

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
            yacoe=("yacoe_play", "sum"),
            first_downs=("is_first_down_earned", "sum"),
            money_down_conv=("is_money_down_conv", "sum"),
            target_epa=("target_epa_play", "sum"),
            non_target_snaps=("is_non_target_snap", "sum"),
            non_target_epa=("non_target_epa_play", "sum"),
            motion_snaps=("is_motion", "sum"),
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
            epa=("epa_clean", "sum"),
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

    # All-Seasons Career Late-Game & Team-Impact Aggregation
    career_summary = (
        player_games_df.groupby("nfl_id")
        .agg(
            career_games=("game_id", "nunique"),
            career_snaps=("snaps", "sum"),
            career_total_routes=("routes", "sum"),
            career_total_targets=("targets", "sum"),
            career_total_catches=("catches", "sum"),
            career_total_yards=("total_yards", "sum"),
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
            career_first_downs=("first_downs", "sum"),
            career_money_down_conv=("money_down_conv", "sum"),
            career_target_epa=("target_epa", "sum"),
            career_non_target_snaps=("non_target_snaps", "sum"),
            career_non_target_epa=("non_target_epa", "sum"),
            career_yacoe=("yacoe", "sum"),
            career_motion_snaps=("motion_snaps", "sum"),
            career_on_field_epa=("epa", "sum"),
        )
        .reset_index()
    )

    q13_routes = career_summary["career_total_routes"] - career_summary["career_q4_routes"]
    q13_yards = career_summary["career_total_yards"] - career_summary["career_q4_yards"]
    career_summary["q13_ypr"] = np.where(q13_routes >= 15, q13_yards / q13_routes, np.nan)
    career_summary["q4_ypr"] = np.where(
        career_summary["career_q4_routes"] >= 10,
        career_summary["career_q4_yards"] / career_summary["career_q4_routes"],
        np.nan,
    )
    career_summary["q4_vs_q13_ypr_diff"] = career_summary["q4_ypr"] - career_summary["q13_ypr"]

    seasons_equiv = np.maximum(1.0, career_summary["career_games"] / 15.0)
    career_summary["clutch_yards_per_game"] = career_summary["career_clutch_yards"] / seasons_equiv
    career_summary["q4_tds_per_game"] = career_summary["career_q4_tds"] / seasons_equiv
    career_summary["q4_explosive_per_game"] = career_summary["career_q4_explosive"] / seasons_equiv
    career_summary["clutch_wpa_per_season"] = career_summary["career_clutch_wpa"] / seasons_equiv

    # Beyond-Box-Score Team Impact & Offensive Gravity Targets
    # 1. Chain-Moving Rate: 1st Downs earned per 100 routes (shrunk slightly toward position mean for tiny sample players)
    rt_denom = np.maximum(25.0, career_summary["career_total_routes"])
    career_summary["first_downs_per_100_routes"] = (career_summary["career_first_downs"] / rt_denom) * 100.0
    # 2. Money-Down Conversions (3rd & 4th down conversions per season)
    career_summary["third_fourth_conv_per_season"] = career_summary["career_money_down_conv"] / seasons_equiv
    # 3. Positive Expected Points Added on Targets per season (EPA+ Created)
    career_summary["target_epa_per_season"] = career_summary["career_target_epa"] / seasons_equiv
    # 4. Target-Earning Power: Targets per 100 Routes Run (TPRR %)
    career_summary["tprr_pct"] = (career_summary["career_total_targets"] / rt_denom) * 100.0
    # 5. Offensive Gravity: Empirical-Bayes shrunk Positive Decoy Team EPA per 100 Non-Targeted Snaps
    career_summary["gravity_non_target_epa_100"] = (
        career_summary["career_non_target_epa"] / (career_summary["career_non_target_snaps"] + 120.0)
    ) * 100.0
    # Additional diagnostic impact metrics
    career_summary["yacoe_per_catch"] = np.where(
        career_summary["career_total_catches"] >= 5,
        career_summary["career_yacoe"] / career_summary["career_total_catches"],
        0.0,
    )
    career_summary["motion_rate_pct"] = (
        career_summary["career_motion_snaps"] / np.maximum(20.0, career_summary["career_snaps"])
    ) * 100.0
    career_summary["on_field_epa_per_100_snaps"] = (
        career_summary["career_on_field_epa"] / np.maximum(30.0, career_summary["career_snaps"])
    ) * 100.0

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
        "career_snaps",
        "career_total_routes",
        "career_total_targets",
        "career_total_catches",
        "career_total_yards",
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
        "first_downs_per_100_routes",
        "third_fourth_conv_per_season",
        "target_epa_per_season",
        "tprr_pct",
        "gravity_non_target_epa_100",
        "yacoe_per_catch",
        "motion_rate_pct",
        "on_field_epa_per_100_snaps",
    ]
    for col in zero_fill_cols:
        df[col] = df[col].fillna(0.0)

    df = df.copy()
    df["yards_per_route"] = np.where(
        df["total_routes"] >= 20, df["total_yards"] / df["total_routes"], np.nan
    )
    df["q4_vs_q13_ypr_diff_filled"] = df["q4_vs_q13_ypr_diff"].fillna(0.0)

    return df, player_games_df, tracking_df


# ---------------------------------------------------------------------------
# Feature Tiers & Target Definitions
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

FEATURE_TIERS = {
    "Tier 1: Pure Combine Physical + Movement Tracking": PHYSICAL_FEATURES + MOVEMENT_FEATURES,
    "Tier 2: Full End-of-Combine Package (Physical + Movement + Combine NGS Scores)": PHYSICAL_FEATURES + MOVEMENT_FEATURES + NGS_PRE_DRAFT_FEATURES,
}

BOX_CLUTCH_TARGETS = [
    "yards_per_game",
    "tds_per_game",
    "clutch_yards_per_game",
    "q4_tds_per_game",
    "q4_explosive_per_game",
    "clutch_wpa_per_season",
]

TEAM_IMPACT_TARGETS = [
    "first_downs_per_100_routes",
    "third_fourth_conv_per_season",
    "target_epa_per_season",
    "tprr_pct",
    "gravity_non_target_epa_100",
]

TARGET_COLUMNS = BOX_CLUTCH_TARGETS + TEAM_IMPACT_TARGETS

# Targets that can take negative values (EPA metrics)
SIGNED_TARGETS = {"target_epa_per_season", "gravity_non_target_epa_100"}
NON_NEGATIVE_MASK = np.array([c not in SIGNED_TARGETS for c in TARGET_COLUMNS], dtype=bool)

TARGET_LABELS = {
    "yards_per_game": "Rookie Yards / Game",
    "tds_per_game": "Rookie TDs / Game",
    "clutch_yards_per_game": "Q4 Clutch Yards / Season",
    "q4_tds_per_game": "4th-Quarter TDs / Season",
    "q4_explosive_per_game": "Q4 Explosive Plays (15+ yd) / Season",
    "clutch_wpa_per_season": "Q4 Clutch Win Prob Added (WPA / Season)",
    "first_downs_per_100_routes": "Chain-Moving: 1st Downs / 100 Routes",
    "third_fourth_conv_per_season": "Money Downs: 3rd/4th Conv / Season",
    "target_epa_per_season": "Value Creation: Target EPA / Season",
    "tprr_pct": "Target-Earning Power: TPRR (%)",
    "gravity_non_target_epa_100": "Offensive Gravity: Decoy Team EPA / 100 Snaps",
}

MODEL_FAMILIES = [
    "Hybrid Stack Ensemble (NN + Bayesian + XGBoost)",
    "Dual-Branch Multi-Task Neural Network",
    "Bayesian Hierarchical Model (Partial Pooling + Credible Intervals)",
    "XGBoost / Gradient Boosted Trees (Native Missingness)",
]


# ---------------------------------------------------------------------------
# MODEL FAMILY 1: Dual-Branch Multi-Task Neural Network
# ---------------------------------------------------------------------------
class DualBranchCombineNN:
    def __init__(
        self,
        phys_indices,
        move_indices,
        n_targets,
        h_phys=16,
        h_move=20,
        h_joint=24,
        lr=0.015,
        weight_decay=0.35,
        epochs=300,
        seed=42,
    ):
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

        self.col_medians = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.input_scale = None
        self.params = {}

    @staticmethod
    def _silu(z):
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
        return z * sig, sig * (1.0 + z * (1.0 - sig))

    def _preprocess_fit(self, X, Y):
        X = np.asarray(X, dtype=float).copy()
        Y = np.asarray(Y, dtype=float).copy()
        n, d = X.shape

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

        corr_mat = np.abs((Xs.T @ Ys) / n)
        input_relevance = np.clip(
            np.max(corr_mat, axis=1) * np.sqrt(np.clip(coverage, 0.2, 1.0)) * 2.5, 0.30, 1.4
        )
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

        self.params = {
            "Wp": rng.normal(0.0, np.sqrt(1.0 / max(1, d_p)), size=(d_p, self.h_phys)),
            "bp": np.zeros(self.h_phys),
            "Wm": rng.normal(0.0, np.sqrt(1.0 / max(1, d_m)), size=(d_m, self.h_move)),
            "bm": np.zeros(self.h_move),
            "Wj": rng.normal(
                0.0,
                np.sqrt(1.0 / (self.h_phys + self.h_move)),
                size=(self.h_phys + self.h_move, self.h_joint),
            ),
            "bj": np.zeros(self.h_joint),
            "Wo": rng.normal(0.0, 0.15 / np.sqrt(self.h_joint), size=(self.h_joint, self.n_targets)),
            "bo": np.zeros(self.n_targets),
            "W_skip": np.linalg.solve(Xs.T @ Xs + 45.0 * np.eye(d), Xs.T @ Ys),
        }

        m_state = {k: np.zeros_like(v) for k, v in self.params.items()}
        v_state = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        delta = 1.25

        Xp = Xs[:, self.phys_idx]
        Xm = Xs[:, self.move_idx]

        for epoch in range(1, self.epochs + 1):
            lr_t = self.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / self.epochs))

            zp = Xp @ self.params["Wp"] + self.params["bp"]
            ap, dap = self._silu(zp)

            zm = Xm @ self.params["Wm"] + self.params["bm"]
            am, dam = self._silu(zm)

            concat_pm = np.concatenate([ap, am], axis=1)
            zj = concat_pm @ self.params["Wj"] + self.params["bj"]
            aj, daj = self._silu(zj)

            y_pred = aj @ self.params["Wo"] + self.params["bo"] + Xs @ self.params["W_skip"]

            err = y_pred - Ys
            grad_out = np.where(np.abs(err) <= delta, err, delta * np.sign(err)) / n

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
        y_pred[:, NON_NEGATIVE_MASK] = np.maximum(0.0, y_pred[:, NON_NEGATIVE_MASK])
        return y_pred


# ---------------------------------------------------------------------------
# MODEL FAMILY 2: Bayesian Hierarchical Model (Partial Pooling + ARD + Credible Intervals)
# ---------------------------------------------------------------------------
class MultiTargetBayesianHierarchicalModel:
    """
    Multi-Target Hierarchical Bayesian Regression featuring:
      1. Position-group partial pooling prior (shrinks position group means toward global prior)
      2. Sample-size / exposure precision weighting (weights players by games/routes played)
      3. Empirical Bayes ARD (Automatic Relevance Determination) feature prior precision
      4. Analytical posterior mean AND posterior predictive standard deviation (for 10th/90th
         percentile Floor & Ceiling credible intervals)
    """

    def __init__(self, alpha_prior=55.0):
        self.alpha_prior = alpha_prior
        self.col_medians = None
        self.x_mean = None
        self.x_std = None
        self.global_means = None
        self.group_priors = {}
        self.post_covs = []
        self.beta_posts = []
        self.sigma2s = []

    def fit(self, X, Y, pos_groups=None, exposure_weights=None):
        X = np.asarray(X, dtype=float).copy()
        Y = np.asarray(Y, dtype=float).copy()
        n, d = X.shape
        n_targets = Y.shape[1]

        if pos_groups is None:
            pos_groups = np.where(X[:, 0] > 0.5, "WR", np.where(X[:, 1] > 0.5, "TE", "RB"))
        if exposure_weights is None:
            exposure_weights = np.ones(n)
        w = np.clip(exposure_weights / np.mean(exposure_weights), 0.25, 2.0)

        coverage = np.mean(np.isfinite(X), axis=0)
        self.col_medians = np.zeros(d)
        for j in range(d):
            fin = X[np.isfinite(X[:, j]), j]
            self.col_medians[j] = float(np.median(fin)) if len(fin) > 0 else 0.0
            X[~np.isfinite(X[:, j]), j] = self.col_medians[j]

        self.x_mean = np.average(X, axis=0, weights=w)
        self.x_std = np.sqrt(np.average((X - self.x_mean) ** 2, axis=0, weights=w))
        self.x_std[self.x_std < 1e-6] = 1.0
        Xs = np.clip((X - self.x_mean) / self.x_std, -3.5, 3.5)

        self.global_means = np.average(Y, axis=0, weights=w)
        self.group_priors = {}
        Y_res = np.zeros_like(Y)
        for g in np.unique(pos_groups):
            mask = pos_groups == g
            n_g = np.sum(w[mask])
            shrink = n_g / (n_g + 8.0)
            g_mean = shrink * np.average(Y[mask], axis=0, weights=w[mask]) + (1.0 - shrink) * self.global_means
            self.group_priors[g] = g_mean
            Y_res[mask] = Y[mask] - g_mean[None, :]

        W_mat = np.diag(w)
        XtWX = Xs.T @ W_mat @ Xs
        self.post_covs = []
        self.beta_posts = []
        self.sigma2s = []

        for t in range(n_targets):
            yt_res = Y_res[:, t]
            sd_y = max(1e-6, float(np.std(yt_res)))
            corrs = np.abs((Xs.T @ (w * yt_res)) / (np.sum(w) * sd_y))
            ard_prec = self.alpha_prior / np.clip(
                (corrs ** 1.2) * np.sqrt(np.clip(coverage, 0.15, 1.0)) * 3.0, 0.20, 2.5
            )
            post_cov = np.linalg.inv(XtWX + np.diag(ard_prec))
            beta_post = post_cov @ (Xs.T @ (w * yt_res))
            prior_vec = np.array([self.group_priors.get(g, self.global_means)[t] for g in pos_groups])
            pred_t = prior_vec + Xs @ beta_post
            sigma2 = float(np.average((Y[:, t] - pred_t) ** 2, weights=w))
            self.post_covs.append(post_cov)
            self.beta_posts.append(beta_post)
            self.sigma2s.append(sigma2)

        return self

    def predict(self, X, pos_groups=None, return_std=False):
        X = np.asarray(X, dtype=float).copy()
        n = len(X)
        if pos_groups is None:
            pos_groups = np.where(X[:, 0] > 0.5, "WR", np.where(X[:, 1] > 0.5, "TE", "RB"))
        for j in range(X.shape[1]):
            X[~np.isfinite(X[:, j]), j] = self.col_medians[j]
        Xs = np.clip((X - self.x_mean) / self.x_std, -3.5, 3.5)

        n_targets = len(self.beta_posts)
        mean_preds = np.zeros((n, n_targets))
        std_preds = np.zeros((n, n_targets))

        for t in range(n_targets):
            prior_vec = np.array([self.group_priors.get(g, self.global_means)[t] for g in pos_groups])
            mean_preds[:, t] = prior_vec + Xs @ self.beta_posts[t]
            if return_std:
                epistemic = np.sum((Xs @ self.post_covs[t]) * Xs, axis=1)
                # Epistemic + aleatoric talent variance across players
                std_preds[:, t] = np.sqrt(np.maximum(1e-4, self.sigma2s[t] * (0.35 + epistemic)))

        mean_preds[:, NON_NEGATIVE_MASK] = np.maximum(0.0, mean_preds[:, NON_NEGATIVE_MASK])
        if return_std:
            return mean_preds, std_preds
        return mean_preds


# ---------------------------------------------------------------------------
# MODEL FAMILY 3: XGBoost-Style Gradient Boosted Decision Trees (Native Missingness)
# ---------------------------------------------------------------------------
class _PureXGBTree:
    def __init__(self, max_depth=2, min_samples_leaf=8, reg_lambda=10.0, colsample=0.65, rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.colsample = colsample
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.tree = None

    def _fit_node(self, X, g, depth):
        n, d = X.shape
        leaf_val = -np.sum(g) / (n + self.reg_lambda)
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return {"is_leaf": True, "val": float(leaf_val)}

        base_score = (np.sum(g) ** 2) / (n + self.reg_lambda)
        best_gain = 0.0
        best_split = None

        feat_indices = self.rng.choice(d, size=max(1, int(d * self.colsample)), replace=False)
        for j in feat_indices:
            col = X[:, j]
            nan_mask = ~np.isfinite(col)
            valid_mask = ~nan_mask
            if np.sum(valid_mask) < 2 * self.min_samples_leaf:
                continue
            v_col = col[valid_mask]
            quantiles = np.unique(np.percentile(v_col, [20, 35, 50, 65, 80]))
            for thresh in quantiles:
                for nan_go_left in (True, False):
                    if nan_go_left:
                        left_m = nan_mask | (col <= thresh)
                    else:
                        left_m = valid_mask & (col <= thresh)
                    n_l = int(np.sum(left_m))
                    n_r = n - n_l
                    if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                        continue
                    g_l = float(np.sum(g[left_m]))
                    g_r = float(np.sum(g[~left_m]))
                    gain = (g_l ** 2) / (n_l + self.reg_lambda) + (g_r ** 2) / (n_r + self.reg_lambda) - base_score
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (j, float(thresh), nan_go_left, left_m)

        if best_split is None or best_gain <= 1e-5:
            return {"is_leaf": True, "val": float(leaf_val)}

        j, thresh, nan_go_left, left_m = best_split
        return {
            "is_leaf": False,
            "feat": j,
            "thresh": thresh,
            "nan_go_left": nan_go_left,
            "left": self._fit_node(X[left_m], g[left_m], depth + 1),
            "right": self._fit_node(X[~left_m], g[~left_m], depth + 1),
        }

    def fit(self, X, g):
        self.tree = self._fit_node(X, g, 0)
        return self

    def _pred_one(self, node, x_row):
        if node["is_leaf"]:
            return node["val"]
        v = x_row[node["feat"]]
        if not np.isfinite(v):
            go_left = node["nan_go_left"]
        else:
            go_left = v <= node["thresh"]
        return self._pred_one(node["left"] if go_left else node["right"], x_row)

    def predict(self, X):
        return np.array([self._pred_one(self.tree, row) for row in X])


class MultiTargetXGBoostRegressor:
    """Multi-target XGBoost-style gradient boosted trees with native missing-value split direction learning."""

    def __init__(self, n_estimators=36, lr=0.065, max_depth=2, min_samples_leaf=8, reg_lambda=10.0, subsample=0.8, colsample=0.65, seed=42):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample = colsample
        self.seed = seed
        self.base_preds = None
        self.target_trees = []

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n, n_targets = Y.shape
        self.base_preds = np.mean(Y, axis=0)
        self.target_trees = []

        for t in range(n_targets):
            y_t = Y[:, t]
            pred = np.full(n, self.base_preds[t])
            rng = np.random.default_rng(self.seed + t * 31)
            trees = []
            for _ in range(self.n_estimators):
                g = pred - y_t
                sub_idx = rng.choice(n, size=int(n * self.subsample), replace=False)
                tree = _PureXGBTree(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    reg_lambda=self.reg_lambda,
                    colsample=self.colsample,
                    rng=rng,
                ).fit(X[sub_idx], g[sub_idx])
                pred += self.lr * tree.predict(X)
                trees.append(tree)
            self.target_trees.append(trees)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        n_targets = len(self.target_trees)
        preds = np.tile(self.base_preds, (n, 1))
        for t, trees in enumerate(self.target_trees):
            for tree in trees:
                preds[:, t] += self.lr * tree.predict(X)
        preds[:, NON_NEGATIVE_MASK] = np.maximum(0.0, preds[:, NON_NEGATIVE_MASK])
        return preds


# ---------------------------------------------------------------------------
# 5-Fold Cross-Validation Training & Multi-Model Evaluation Suite
# ---------------------------------------------------------------------------
def train_and_evaluate_tiers(df, k=5, seed=42):
    """
    Trains 5-Fold Cross-Validated ensembles of:
      - Dual-Branch Multi-Task Neural Network
      - Bayesian Hierarchical Model (with analytical 10th-90th pct Floor/Ceiling intervals)
      - XGBoost / Gradient Boosted Decision Trees
      - Hybrid Stack Ensemble (0.45 NN + 0.35 Bayesian + 0.20 XGBoost)
    across all 3 Feature Tiers and all 11 Targets.
    """
    results = {}
    Y = df[TARGET_COLUMNS].to_numpy(dtype=float)
    pos_all = df["nfl_position"].values
    exp_all = np.maximum(1.0, df["career_games"].values)
    n = len(df)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    for tier_name, feat_cols in FEATURE_TIERS.items():
        X_df = df[feat_cols]
        X_raw = X_df.to_numpy(dtype=float, na_value=np.nan)
        phys_idx = [i for i, c in enumerate(feat_cols) if c not in MOVEMENT_FEATURES]
        move_idx = [i for i, c in enumerate(feat_cols) if c in MOVEMENT_FEATURES]

        oof_nn = np.zeros_like(Y)
        oof_bayes = np.zeros_like(Y)
        oof_bayes_std = np.zeros_like(Y)
        oof_xgb = np.zeros_like(Y)

        nn_models = []
        bayes_models = []
        xgb_models = []

        for f_i, val_idx in enumerate(folds):
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False

            # 1. Dual-Branch Multi-Task NN
            nn_m = DualBranchCombineNN(
                phys_indices=phys_idx,
                move_indices=move_idx,
                n_targets=len(TARGET_COLUMNS),
                seed=seed + f_i * 17,
            ).fit(X_raw[train_mask], Y[train_mask])
            oof_nn[val_idx] = nn_m.predict(X_raw[val_idx])
            nn_models.append(nn_m)

            # 2. Bayesian Hierarchical Model
            bay_m = MultiTargetBayesianHierarchicalModel(alpha_prior=55.0).fit(
                X_raw[train_mask],
                Y[train_mask],
                pos_groups=pos_all[train_mask],
                exposure_weights=exp_all[train_mask],
            )
            b_mean, b_std = bay_m.predict(X_raw[val_idx], pos_groups=pos_all[val_idx], return_std=True)
            oof_bayes[val_idx] = b_mean
            oof_bayes_std[val_idx] = b_std
            bayes_models.append(bay_m)

            # 3. XGBoost / GBDT
            xgb_m = MultiTargetXGBoostRegressor(seed=seed + f_i * 13).fit(X_raw[train_mask], Y[train_mask])
            oof_xgb[val_idx] = xgb_m.predict(X_raw[val_idx])
            xgb_models.append(xgb_m)

        # 4. Hybrid Stack Ensemble
        oof_hybrid = 0.45 * oof_nn + 0.35 * oof_bayes + 0.20 * oof_xgb

        oof_by_family = {
            "Hybrid Stack Ensemble (NN + Bayesian + XGBoost)": oof_hybrid,
            "Dual-Branch Multi-Task Neural Network": oof_nn,
            "Bayesian Hierarchical Model (Partial Pooling + Credible Intervals)": oof_bayes,
            "XGBoost / Gradient Boosted Trees (Native Missingness)": oof_xgb,
        }

        all_fold_nn = np.stack([m.predict(X_raw) for m in nn_models], axis=0)
        ensemble_std = all_fold_nn.std(axis=0)

        # Compute 10th-percentile Floor and 90th-percentile Ceiling from Bayesian Posterior Predictive Distribution
        # (z = ±1.2816 corresponds to 10th and 90th percentiles)
        bayes_floor_10 = oof_bayes - 1.2816 * oof_bayes_std
        bayes_floor_10[:, NON_NEGATIVE_MASK] = np.maximum(0.0, bayes_floor_10[:, NON_NEGATIVE_MASK])
        bayes_ceil_90 = oof_bayes + 1.2816 * oof_bayes_std

        # Metrics across all 4 model families
        metrics_by_family = {}
        benchmark_rows = []
        for fam_name, oof_mat in oof_by_family.items():
            fam_metrics = []
            for t_i, t_col in enumerate(TARGET_COLUMNS):
                y_true = Y[:, t_i]
                y_hat = oof_mat[:, t_i]
                r, p = pearson_r_pval(y_true, y_hat)
                rho, _ = spearman_rho_pval(y_true, y_hat)
                mae = float(np.mean(np.abs(y_true - y_hat)))
                rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
                item = {
                    "model_family": fam_name,
                    "target": t_col,
                    "target_label": TARGET_LABELS[t_col],
                    "r": r,
                    "p_val": p,
                    "spearman_rho": rho,
                    "mae": mae,
                    "rmse": rmse,
                }
                fam_metrics.append(item)
                benchmark_rows.append(item)
            metrics_by_family[fam_name] = pd.DataFrame(fam_metrics)

        importances = compute_feature_importances(nn_models, X_df, feat_cols)

        results[tier_name] = {
            "feature_cols": feat_cols,
            "models": nn_models,
            "nn_models": nn_models,
            "bayes_models": bayes_models,
            "xgb_models": xgb_models,
            "oof_preds": oof_hybrid,
            "oof_by_family": oof_by_family,
            "ensemble_std": ensemble_std,
            "bayes_std": oof_bayes_std,
            "bayes_floor_10": bayes_floor_10,
            "bayes_ceil_90": bayes_ceil_90,
            "metrics_df": metrics_by_family["Hybrid Stack Ensemble (NN + Bayesian + XGBoost)"],
            "metrics_by_family": metrics_by_family,
            "benchmark_df": pd.DataFrame(benchmark_rows),
            "importances_df": importances,
        }

    return results


def compute_feature_importances(models, X_df, feat_cols):
    X_raw = X_df.to_numpy(dtype=float, na_value=np.nan)
    rows = []

    for j, col in enumerate(feat_cols):
        col_vals = X_raw[:, j]
        sd = np.nanstd(col_vals)
        if not np.isfinite(sd) or sd < 1e-6:
            continue
        X_plus = X_raw.copy()
        X_minus = X_raw.copy()
        X_plus[:, j] = col_vals + 0.5 * sd
        X_minus[:, j] = col_vals - 0.5 * sd

        pred_plus = np.mean([m.predict(X_plus) for m in models], axis=0)
        pred_minus = np.mean([m.predict(X_minus) for m in models], axis=0)
        delta = np.mean(pred_plus - pred_minus, axis=0)

        cat = (
            "Movement Tracking"
            if col in MOVEMENT_FEATURES
            else ("NGS Combine Score" if col in NGS_PRE_DRAFT_FEATURES else "Combine Physical")
        )
        row = {"feature": col, "category": cat}
        for t_i, t_col in enumerate(TARGET_COLUMNS):
            row[f"effect_{t_col}"] = float(delta[t_i])
            row[f"abs_effect_{t_col}"] = float(abs(delta[t_i]))
        rows.append(row)

    return pd.DataFrame(rows)


def predict_custom_prospect(tier_res_Or_models, feat_cols, custom_feature_dict):
    """
    Runs a single prospect's feature dictionary through the active tier's model suite,
    returning predictions across all 4 model families plus Bayesian 10th/90th credible intervals.
    """
    x_vec = np.array([[custom_feature_dict.get(c, np.nan) for c in feat_cols]], dtype=float)

    if isinstance(tier_res_Or_models, dict) and "nn_models" in tier_res_Or_models:
        nn_models = tier_res_Or_models["nn_models"]
        bayes_models = tier_res_Or_models["bayes_models"]
        xgb_models = tier_res_Or_models["xgb_models"]

        nn_folds = np.stack([m.predict(x_vec)[0] for m in nn_models], axis=0)
        nn_mean = nn_folds.mean(axis=0)
        nn_std = nn_folds.std(axis=0)

        bay_outs = [m.predict(x_vec, return_std=True) for m in bayes_models]
        bay_mean = np.mean([o[0][0] for o in bay_outs], axis=0)
        bay_std = np.mean([o[1][0] for o in bay_outs], axis=0)

        xgb_mean = np.mean([m.predict(x_vec)[0] for m in xgb_models], axis=0)
        hyb_mean = 0.45 * nn_mean + 0.35 * bay_mean + 0.20 * xgb_mean

        out = {}
        for i, t_col in enumerate(TARGET_COLUMNS):
            fl = float(bay_mean[i] - 1.2816 * bay_std[i])
            if NON_NEGATIVE_MASK[i]:
                fl = max(0.0, fl)
            cl = float(bay_mean[i] + 1.2816 * bay_std[i])
            out[t_col] = {
                "mean": float(hyb_mean[i]),
                "std": float(nn_std[i]),
                "nn": float(nn_mean[i]),
                "bayesian": float(bay_mean[i]),
                "xgb": float(xgb_mean[i]),
                "hybrid": float(hyb_mean[i]),
                "floor_10": fl,
                "ceil_90": cl,
            }
        return out

    # Fallback if a plain list of NN models was passed
    models = tier_res_Or_models
    fold_preds = np.stack([m.predict(x_vec)[0] for m in models], axis=0)
    mean_pred = fold_preds.mean(axis=0)
    std_pred = fold_preds.std(axis=0)
    return {
        t_col: {"mean": float(mean_pred[i]), "std": float(std_pred[i])}
        for i, t_col in enumerate(TARGET_COLUMNS)
    }


def compute_bayesian_season_trajectory(prior_ypg, prior_std, player_games_df, tau_prior_games=6.5):
    """
    Computes week-by-week Bayesian conjugate Normal-Normal updating of a player's
    True Yards/Game talent estimate from Week 0 (100% Combine Prior) through each game played:
      theta_post(k) = (tau_0^-2 * mu_0 + k * sigma^-2 * y_bar_k) / (tau_0^-2 + k * sigma^-2)
      equivalent to shrinkage weight w_k = k / (k + tau_prior_games) on observed NFL games.
    Returns a DataFrame with Week 0..K posterior mean, 10th-pct floor, 90th-pct ceiling,
    and cumulative observed average.
    """
    prior_ypg = float(max(2.0, prior_ypg))
    prior_std = float(max(4.0, prior_std))

    rows = [
        {
            "game_num": 0,
            "game_label": "Pre-Draft (Combine Prior)",
            "posterior_mean": prior_ypg,
            "floor_10": max(0.0, prior_ypg - 1.2816 * prior_std),
            "ceil_90": prior_ypg + 1.2816 * prior_std,
            "cum_observed_avg": np.nan,
            "game_actual_yards": np.nan,
            "prior_weight_pct": 100.0,
        }
    ]

    if player_games_df is None or len(player_games_df) == 0:
        return pd.DataFrame(rows)

    g_sorted = player_games_df.sort_values(["season", "week"]).reset_index(drop=True)
    cum_yards = 0.0
    cum_route_equiv = 0.0

    for idx, gr in g_sorted.iterrows():
        k = idx + 1
        g_yds = float(gr["total_yards"])
        # Weight each game's evidence by routes run relative to a standard 18-route game
        route_eq = max(0.25, float(gr["routes"]) / 18.0)
        cum_yards += g_yds
        cum_route_equiv += route_eq
        obs_avg = cum_yards / k

        # Empirical Bayes precision weight
        w_obs = cum_route_equiv / (cum_route_equiv + tau_prior_games)
        post_mean = (1.0 - w_obs) * prior_ypg + w_obs * obs_avg
        post_std = prior_std * math.sqrt(1.0 - 0.65 * w_obs)

        lbl = f"{int(gr['season'])} W{int(gr['week'])} vs {gr['opponent_abbr']}"
        rows.append(
            {
                "game_num": k,
                "game_label": lbl,
                "posterior_mean": float(post_mean),
                "floor_10": float(max(0.0, post_mean - 1.2816 * post_std)),
                "ceil_90": float(post_mean + 1.2816 * post_std),
                "cum_observed_avg": float(obs_avg),
                "game_actual_yards": g_yds,
                "prior_weight_pct": float((1.0 - w_obs) * 100.0),
            }
        )

    return pd.DataFrame(rows)


def project_player_game(
    player_row,
    oof_pred_dict,
    routes_in_game=20,
    game_situation="Standard Full Game (Q1-Q4)",
    is_clutch_close=False,
):
    """
    Conditions a player's Combine model projection on specific game opportunity (routes_in_game)
    and late-game situation, returning projected yards, TDs, chain-moving 1st downs, target EPA,
    and Win Probability Added.
    """
    base_ypg = max(2.0, float(oof_pred_dict["yards_per_game"]))
    base_tdpg = max(0.01, float(oof_pred_dict["tds_per_game"]))
    clutch_ypg = max(1.0, float(oof_pred_dict["clutch_yards_per_game"]))
    q4_tdpg = max(0.005, float(oof_pred_dict["q4_tds_per_game"]))
    q4_exp_pg = max(0.01, float(oof_pred_dict["q4_explosive_per_game"]))
    fd_per_100 = max(2.0, float(oof_pred_dict.get("first_downs_per_100_routes", 11.5)))
    tprr_pct = max(5.0, float(oof_pred_dict.get("tprr_pct", 18.0)))
    delta_ypr = float(player_row.get("q4_vs_q13_ypr_diff_filled", 0.0)) if player_row is not None else 0.0

    ref_routes = 16.5
    opp_multiplier = routes_in_game / ref_routes

    if game_situation == "4th Quarter / Overtime Only":
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
        clutch_boost = 1.0 + (0.08 * np.clip(delta_ypr, -0.5, 1.0) if is_clutch_close else 0.0)
        proj_yards = base_ypg * opp_multiplier * clutch_boost
        proj_tds = base_tdpg * opp_multiplier * (1.12 if is_clutch_close else 1.0)
        explosive_prob = 1.0 - math.exp(-max(0.05, (proj_yards / 28.0)))
        expected_wpa = proj_tds * (0.22 if is_clutch_close else 0.14) + (proj_yards / 100.0) * 0.08

    proj_first_downs = (fd_per_100 / 100.0) * routes_in_game * (1.08 if is_clutch_close else 1.0)
    proj_targets = (tprr_pct / 100.0) * routes_in_game
    prob_1plus_td = 1.0 - math.exp(-proj_tds)

    return {
        "projected_yards": float(proj_yards),
        "projected_tds": float(proj_tds),
        "projected_targets": float(proj_targets),
        "projected_first_downs": float(proj_first_downs),
        "prob_1plus_td": float(prob_1plus_td),
        "prob_explosive_15yd": float(explosive_prob),
        "expected_wpa": float(expected_wpa),
    }


if __name__ == "__main__":
    print("Building Skill Position Combine, Movement & Team-Impact dataset from DuckDB...")
    df, player_games_df, _ = build_skill_position_dataset()
    print(f"Loaded {len(df)} skill-position players and {len(player_games_df)} player-games.")

    tier_results = train_and_evaluate_tiers(df, k=5, seed=42)
    t2 = tier_results["Tier 2: Full End-of-Combine Package (Physical + Movement + Combine NGS Scores)"]
    print("\n" + "=" * 95)
    print("5-FOLD CV BENCHMARK (Tier 2: Full End-of-Combine Package — Hybrid Stack Ensemble)")
    print("=" * 95)
    m_df = t2["metrics_df"].copy()
    m_df["r"] = m_df["r"].round(3)
    m_df["p_val"] = m_df["p_val"].round(4)
    m_df["spearman_rho"] = m_df["spearman_rho"].round(3)
    m_df["mae"] = m_df["mae"].round(3)
    print(m_df[["target_label", "r", "p_val", "spearman_rho", "mae"]].to_string(index=False))
