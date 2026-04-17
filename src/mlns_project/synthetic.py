from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def generate_synthetic_dataset(
    markets_path: Path,
    observations_path: Path,
    seed: int = 42,
    n_days: int = 90,
    strikes_per_type: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    assets = ["BTC", "ETH"]
    contract_types = ["above", "below"]
    start_time = pd.Timestamp("2026-01-01T12:00:00Z")
    observation_offsets = list(range(48, 0, -6))

    market_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []

    for day_index in range(n_days):
        settlement_time = start_time + pd.Timedelta(days=day_index)

        for asset in assets:
            base_level = 42000 if asset == "BTC" else 2400
            seasonal = 0.06 * np.sin(day_index / 11.0)
            trend = 0.0008 * day_index
            settlement_spot = base_level * (1.0 + seasonal + trend + rng.normal(0.0, 0.025))
            cohort_noise = rng.normal(0.0, 0.18)

            for contract_type in contract_types:
                strike_offsets = np.linspace(-0.12, 0.12, strikes_per_type)
                for strike_index, strike_offset in enumerate(strike_offsets):
                    market_id = f"{asset}-{settlement_time:%Y%m%d}-{contract_type}-{strike_index}"
                    strike = settlement_spot * (1.0 + strike_offset + rng.normal(0.0, 0.01))

                    if contract_type == "above":
                        label = int(settlement_spot > strike)
                    else:
                        label = int(settlement_spot < strike)

                    market_rows.append(
                        {
                            "market_id": market_id,
                            "asset": asset,
                            "contract_type": contract_type,
                            "strike": round(float(strike), 4),
                            "settlement_time": settlement_time.isoformat(),
                            "label": label,
                        }
                    )

                    local_bias = 0.05 * np.tanh((strike_offset * 10.0) + cohort_noise)
                    for hours_before in observation_offsets:
                        timestamp = settlement_time - pd.Timedelta(hours=hours_before)
                        time_fraction = 1.0 - (hours_before / max(observation_offsets))
                        drift = 0.04 * np.sin((day_index + hours_before) / 8.0)
                        reference_spot = settlement_spot * (
                            1.0
                            - 0.025 * (1.0 - time_fraction)
                            + 0.01 * drift
                            + rng.normal(0.0, 0.008)
                        )
                        signed_gap = (strike - reference_spot) / reference_spot

                        if contract_type == "above":
                            latent_probability = _sigmoid(-12.0 * signed_gap)
                        else:
                            latent_probability = _sigmoid(12.0 * signed_gap)

                        mid_price = np.clip(
                            latent_probability + local_bias + rng.normal(0.0, 0.03),
                            0.01,
                            0.99,
                        )
                        spread = max(0.002, 0.02 + 0.04 * abs(signed_gap) + rng.normal(0.0, 0.004))
                        liquidity = max(50.0, 1800.0 * (1.0 - abs(signed_gap)) + rng.normal(0.0, 120.0))
                        volume = max(10.0, 450.0 * (1.0 + time_fraction) + rng.normal(0.0, 40.0))

                        observation_rows.append(
                            {
                                "market_id": market_id,
                                "timestamp": timestamp.isoformat(),
                                "mid_price": round(float(mid_price), 6),
                                "spread": round(float(spread), 6),
                                "liquidity": round(float(liquidity), 6),
                                "volume": round(float(volume), 6),
                                "reference_spot_price": round(float(reference_spot), 6),
                            }
                        )

    markets = pd.DataFrame(market_rows)
    observations = pd.DataFrame(observation_rows)

    markets_path.parent.mkdir(parents=True, exist_ok=True)
    observations_path.parent.mkdir(parents=True, exist_ok=True)
    markets.to_csv(markets_path, index=False)
    observations.to_csv(observations_path, index=False)
    return markets, observations

