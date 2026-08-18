"""Central pydantic-settings for CUAD."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/cuad/config/settings.py → project root is parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE.is_file() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    project_root: Path = Field(default=PROJECT_ROOT)
    runs_dir: str = Field(default="runs", validation_alias="PANELCAST_ORCH_RUNS_DIR")
    saved_models_dir: str = Field(default="saved_models", validation_alias="PANELCAST_AGENTS_SAVE_DIR")

    data_region: str = Field(default="ISO-NE", validation_alias="PANELCAST_DATA_REGION")
    weather_provider: str = Field(default="open_meteo", validation_alias="PANELCAST_DATA_WEATHER_PROVIDER")
    weather_station: str = Field(default="BOS", validation_alias="PANELCAST_DATA_WEATHER_STATION")
    lags: str = Field(default="24,48,168", validation_alias="PANELCAST_DATA_LAGS")
    roll_windows: str = Field(default="24", validation_alias="PANELCAST_DATA_ROLL_WINDOWS")
    train_frac: float = Field(default=0.7, validation_alias="PANELCAST_DATA_TRAIN_FRAC")
    cal_frac: float = Field(default=0.15, validation_alias="PANELCAST_DATA_CAL_FRAC")
    cache_enabled: bool = Field(default=True, validation_alias="PANELCAST_DATA_CACHE_ENABLED")
    use_synthetic_if_missing: bool = Field(
        default=False, validation_alias="PANELCAST_DATA_USE_SYNTHETIC_IF_MISSING"
    )

    iso_ne_username: str = Field(default="", validation_alias="ISO_NE_USERNAME")
    iso_ne_password: str = Field(default="", validation_alias="ISO_NE_PASSWORD")
    noaa_token: str = Field(default="", validation_alias="NOAA_TOKEN")
    eia_api_key: str = Field(default="", validation_alias="EIA_API_KEY")
    acn_api_token: str = Field(default="", validation_alias="ACN_API_TOKEN")

    panel: str = Field(
        default="ridge,lightgbm,lstm,foundation_ts",
        validation_alias="PANELCAST_AGENTS_PANEL",
    )
    foundation_enabled: bool = Field(default=True, validation_alias="PANELCAST_AGENTS_FOUNDATION_ENABLED")
    lstm_lookback: int = Field(default=168, validation_alias="PANELCAST_AGENTS_LSTM_LOOKBACK")
    use_tuned_params: bool = Field(default=False, validation_alias="PANELCAST_AGENTS_USE_TUNED_PARAMS")

    cv_n_folds: int = Field(default=5, validation_alias="PANELCAST_CV_N_FOLDS")
    cv_min_train_rows: int = Field(default=4032, validation_alias="PANELCAST_CV_MIN_TRAIN_ROWS")
    cv_val_rows: int = Field(default=168, validation_alias="PANELCAST_CV_VAL_ROWS")
    cv_horizon: int = Field(default=24, validation_alias="PANELCAST_CV_HORIZON")
    cv_step: int = Field(default=24, validation_alias="PANELCAST_CV_STEP")
    cv_tune_metric: str = Field(default="MAPE", validation_alias="PANELCAST_CV_TUNE_METRIC")

    alpha: float = Field(default=0.1, validation_alias="PANELCAST_UNCERTAINTY_ALPHA")
    combine_method: str = Field(default="mean", validation_alias="PANELCAST_UNCERTAINTY_COMBINE")
    spread_metric: str = Field(default="std", validation_alias="PANELCAST_UNCERTAINTY_SPREAD_METRIC")
    conformal_mode: str = Field(default="split", validation_alias="PANELCAST_UNCERTAINTY_MODE")
    aci_gamma: float = Field(default=0.005, validation_alias="PANELCAST_UNCERTAINTY_ACI_GAMMA")
    spread_floor: float = Field(default=1e-3, validation_alias="PANELCAST_UNCERTAINTY_SPREAD_FLOOR")
    two_sided_conformal: bool = Field(default=False, validation_alias="PANELCAST_UNCERTAINTY_TWO_SIDED")
    calibration_max_steps: int = Field(default=50, validation_alias="PANELCAST_UNCERTAINTY_CAL_MAX_STEPS")
    fast_calibration: bool = Field(default=True, validation_alias="PANELCAST_UNCERTAINTY_FAST_CALIBRATION")

    gate_enabled: bool = Field(default=False, validation_alias="PANELCAST_ORCH_GATE_ENABLED")
    parallel_agents: bool = Field(default=True, validation_alias="PANELCAST_ORCH_PARALLEL_AGENTS")

    api_port: int = Field(default=8000, validation_alias="PANELCAST_IF_API_PORT")
    mcp_enabled: bool = Field(default=False, validation_alias="PANELCAST_IF_MCP_ENABLED")

    # Reproducibility
    seed: int = Field(default=42, validation_alias="PANELCAST_SEED")

    # Rolling-origin evaluation stride (hours between forecast origins in test set)
    eval_roll_step: int = Field(default=24, validation_alias="PANELCAST_EVAL_ROLL_STEP")

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def region_processed_dir(self) -> Path:
        return self.processed_dir / self.data_region

    @property
    def features_path(self) -> Path:
        return self.region_processed_dir / "features.parquet"

    @property
    def splits_path(self) -> Path:
        return self.region_processed_dir / "splits.json"

    @property
    def runs_path(self) -> Path:
        return self.project_root / self.runs_dir

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def best_params_path(self) -> Path:
        return self.results_dir / "best_params.json"

    def lag_hours(self) -> list[int]:
        return [int(x.strip()) for x in self.lags.split(",") if x.strip()]

    def roll_hour_windows(self) -> list[int]:
        return [int(x.strip()) for x in self.roll_windows.split(",") if x.strip()]

    def panel_names(self) -> list[str]:
        return [x.strip() for x in self.panel.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
