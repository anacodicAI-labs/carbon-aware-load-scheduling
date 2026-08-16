"""Lifecycle greenhouse-gas emission factors by EIA fuel type code.

Values are gCO2-equivalent per kWh, lifecycle (not just combustion).

Provenance: SIX of the eight values are IPCC AR5 medians; TWO (OIL, OTH) are
not in AR5 and are documented separately. Do not describe this whole table as
"AR5 emission factors" -- that is accurate for six codes only.

AR5 source for the six: IPCC AR5, WG3 (2014), Annex III, Table A.III.2, median
lifecycle GHG emissions of electricity generation technologies, quoted below
with the AR5 (min / median / max) range where one is given.
    https://www.ipcc.ch/report/ar5/wg3/  (Annex III, Table A.III.2)

Per-code source (the numeric value is fixed; only the attribution is being made
precise -- no value is being changed):
    COL 820  AR5 coal-PC median            (AR5 range 740 / 820 / 910).
    NG  490  AR5 gas combined-cycle median (AR5 range 410 / 490 / 650).
    NUC  12  AR5 nuclear median.
    SUN  48  AR5 utility-scale solar PV median. This is the AR5 utility value,
             NOT Carbon Explorer's 41 for rooftop PV, so the table is not a
             straight copy of Carbon Explorer's.
    WAT  24  AR5 hydropower median.
    WND  11  AR5 onshore wind median.
    OIL 650  NOT AR5: AR5 Annex III Table A.III.2 has no oil row at all. 650 is
             the lifecycle oil value in Carbon Explorer (Acun et al., ASPLOS '23,
             DOI 10.1145/3575693.3575754, Table 2) and, independently, CarbonCast
             (Maji et al., BuildSys '22, DOI 10.1145/3563357.3564079, Table 1:
             650 lifecycle / 406 direct). It is NOT an IPCC SRREN value.
    OTH 230  NOT a published factor for this bucket: it is the AR5 dedicated-
             biomass median (AR5 range 130 / 230 / 420) used as a PROXY for EIA's
             blended "other" category (refuse / biomass / landfill gas). This is
             our modeling choice. CarbonCast treats "Other" (700) and "Biomass"
             (230) as SEPARATE categories; we map EIA OTH to the biomass proxy.
             Case for 230 over 700, EIA "other" composition, and the 130-420
             sensitivity: see notes/emission-factor-provenance.md.

Units: EIA generation is in MWh and these factors are per kWh. The mismatch
CANCELS in the intensity ratio -- carbon_intensity() computes the generation-
weighted mean sum(gen_mwh * factor) / sum(gen_mwh), so the MWh unit divides out
and the result is gCO2/kWh. Do not "fix" the units by rescaling the factors.
"""
from __future__ import annotations

# EIA fuel type code -> lifecycle emission factor (gCO2eq/kWh). Per-value source
# tagged inline (AR5 = IPCC AR5 Annex III Table A.III.2); see the module
# docstring and notes/emission-factor-provenance.md for full attribution.
FACTORS: dict[str, float] = {
    "COL": 820.0,  # AR5 coal-PC median
    "NG": 490.0,   # AR5 gas combined-cycle median
    "NUC": 12.0,   # AR5 nuclear median
    "OIL": 650.0,  # NOT AR5: Carbon Explorer / CarbonCast lifecycle oil
    "SUN": 48.0,   # AR5 utility-scale solar PV median (not rooftop 41)
    "WAT": 24.0,   # AR5 hydropower median
    "WND": 11.0,   # AR5 onshore wind median
    "OTH": 230.0,  # NOT AR5 for this bucket: AR5 biomass median used as proxy
}

# Fallback code for unknown / blended fuel types.
OTHER_CODE: str = "OTH"
