"""Sector-aware PESTEL context shared by the market explorer and deep evidence.

Each sector carries a current read and a forward watch item for the six PESTEL
factors. The entries are deliberately general (policy, rates, demand,
technology, climate, and legal exposure that recur for the sector) so the
deterministic fallback still gives users a real starting point; provider
interpretation adds instrument-specific, time-sensitive detail on top.
"""

from __future__ import annotations

from typing import Any

PESTEL_FACTORS = ("Political", "Economic", "Social", "Technological", "Environmental", "Legal")

SECTOR_PESTEL: dict[str, dict[str, tuple[str, str]]] = {
    "Technology": {
        "Political": ("Export controls, semiconductor subsidies, and platform-regulation politics shape supply chains and market access.", "Watch chip export rules, digital-services taxes, and government AI procurement."),
        "Economic": ("Valuations are rate-sensitive; enterprise IT and advertising budgets follow the business cycle.", "Track rate expectations, capex guidance from hyperscalers, and ad-spend surveys."),
        "Social": ("AI adoption, privacy expectations, and remote-work habits drive product demand.", "Watch consumer trust in AI products and enterprise adoption curves."),
        "Technological": ("AI infrastructure, cloud migration, and rapid product cycles reset competitive positions quickly.", "Monitor model-cost declines, new hardware generations, and platform shifts."),
        "Environmental": ("Data-centre power and water use are under scrutiny; energy costs affect margins.", "Watch grid constraints for new capacity and sustainability disclosure rules."),
        "Legal": ("Antitrust cases, data-privacy law, and IP litigation can change business models.", "Recheck pending antitrust remedies and privacy enforcement before acting."),
    },
    "Financial Services": {
        "Political": ("Capital rules, housing policy, and fiscal stance influence lending volumes and provisioning.", "Watch bank-capital reforms, mortgage rules, and election-cycle banking politics."),
        "Economic": ("Net interest margins, credit losses, and capital-markets activity follow rates and growth.", "Track the yield curve, unemployment, and consumer-credit delinquency trends."),
        "Social": ("Digital banking adoption and demographic wealth transfer reshape deposits and advice.", "Watch deposit migration to fintech and retirement-product demand."),
        "Technological": ("Fintech competition, payments modernization, and AI in underwriting change cost structures.", "Monitor open-banking rollouts and fraud/cyber incidents."),
        "Environmental": ("Climate risk feeds insurance losses, collateral values, and lending policies.", "Watch catastrophe-loss seasons and climate-stress-test outcomes."),
        "Legal": ("Consumer-protection enforcement, sanctions compliance, and litigation drive one-off charges.", "Recheck regulatory investigations and settlement timelines."),
    },
    "Healthcare": {
        "Political": ("Drug-pricing negotiation, public-health budgets, and approval-agency priorities shape revenue.", "Watch price-negotiation lists, coverage rules, and agency leadership changes."),
        "Economic": ("Demand is defensive, but reimbursement rates and hospital budgets follow fiscal cycles.", "Track reimbursement updates and utilization trends."),
        "Social": ("Ageing populations and chronic-disease prevalence support structural demand.", "Watch adoption of new therapy classes and patient-access programs."),
        "Technological": ("Biotech pipelines, gene and obesity therapies, and AI diagnostics drive winners and losers.", "Monitor trial readouts and approval calendars."),
        "Environmental": ("Supply-chain resilience and heat-related health demand are rising considerations.", "Watch manufacturing disruptions and extreme-weather health impacts."),
        "Legal": ("Patent cliffs, product-liability suits, and antitrust in pharmacy benefits can move valuations.", "Recheck patent expiry dates and active litigation."),
    },
    "Energy": {
        "Political": ("OPEC+ decisions, sanctions, pipeline approvals, and carbon policy set the operating backdrop.", "Watch production quotas, permitting decisions, and carbon-price changes."),
        "Economic": ("Commodity prices, refining margins, and global demand growth dominate earnings.", "Track crude and gas curves, inventories, and demand forecasts."),
        "Social": ("Public pressure on emissions coexists with affordability and energy-security concerns.", "Watch shifts in public support for new projects."),
        "Technological": ("Efficiency gains, LNG capacity, and low-carbon investments change long-run cash flows.", "Monitor project sanctioning and cost curves for transition assets."),
        "Environmental": ("Emissions regulation, spill risk, and weather-driven demand are direct exposures.", "Watch methane rules, hurricane season, and transition-capex plans."),
        "Legal": ("Climate litigation, royalty disputes, and regulatory fines are recurring risks.", "Recheck pending environmental cases and regulatory reviews."),
    },
    "Industrials": {
        "Political": ("Infrastructure spending, defence budgets, and tariffs drive order books.", "Watch stimulus programs, procurement awards, and trade-policy shifts."),
        "Economic": ("Cyclical demand follows PMI readings, capex cycles, and freight volumes.", "Track manufacturing PMIs, rail/air traffic, and backlog conversion."),
        "Social": ("Reshoring and skilled-labour shortages influence costs and capacity.", "Watch labour negotiations and wage inflation."),
        "Technological": ("Automation, electrification, and aerospace technology cycles reshape margins.", "Monitor new-platform delivery schedules and automation adoption."),
        "Environmental": ("Emission standards and climate-resilience spending create both cost and demand.", "Watch fleet-emission rules and grid/water infrastructure programs."),
        "Legal": ("Safety incidents, product recalls, and export-compliance cases carry large penalties.", "Recheck safety investigations and compliance findings."),
    },
    "Consumer Defensive": {
        "Political": ("Food, tobacco, and alcohol policy plus trade rules affect pricing and sourcing.", "Watch tariff changes on inputs and health-tax proposals."),
        "Economic": ("Demand is stable, but input inflation and private-label competition squeeze margins.", "Track grocery inflation, wage growth, and promotional intensity."),
        "Social": ("Health-conscious and value-seeking consumers shift category mix.", "Watch GLP-1-driven consumption changes and discount-channel share."),
        "Technological": ("Retail media, supply-chain analytics, and e-commerce logistics are competitive levers.", "Monitor automation and digital-loyalty rollouts."),
        "Environmental": ("Crop yields, packaging rules, and water stress affect costs.", "Watch harvest forecasts and packaging-regulation deadlines."),
        "Legal": ("Labelling, antitrust in grocery, and product-safety recalls are recurring exposures.", "Recheck competition reviews and recall notices."),
    },
    "Consumer Cyclical": {
        "Political": ("Tariffs, EV incentives, and consumer-credit rules move demand and cost of goods.", "Watch trade negotiations and subsidy program changes."),
        "Economic": ("Discretionary demand is highly sensitive to employment, rates, and confidence.", "Track consumer confidence, credit availability, and inventory levels."),
        "Social": ("Brand relevance, experience spending, and demographic tastes drive share.", "Watch spending shifts between goods and experiences."),
        "Technological": ("E-commerce, autonomous driving, and AI personalization change unit economics.", "Monitor digital-sales mix and technology launch timelines."),
        "Environmental": ("Sustainable sourcing and circular-economy rules add cost but also brand value.", "Watch textile, packaging, and vehicle-emission regulations."),
        "Legal": ("Franchise, labour, and product-safety law can hit margins quickly.", "Recheck labour rulings and safety recalls."),
    },
    "Utilities": {
        "Political": ("Rate-case outcomes and energy policy define allowed returns.", "Watch regulator decisions and clean-energy mandates."),
        "Economic": ("Rate-sensitive valuations; capex plans depend on financing costs.", "Track bond yields and rate-base growth guidance."),
        "Social": ("Affordability pressure and reliability expectations shape political support.", "Watch customer-bill politics and outage performance."),
        "Technological": ("Grid modernization, storage, and data-centre load growth drive investment.", "Monitor interconnection queues and load-growth forecasts."),
        "Environmental": ("Wildfire, storm, and decarbonization mandates create liabilities and capex.", "Watch wildfire seasons and coal-retirement timelines."),
        "Legal": ("Liability for outages or fires and regulatory compliance are material.", "Recheck pending liability cases and compliance orders."),
    },
    "Communication Services": {
        "Political": ("Spectrum policy, content regulation, and platform politics matter for access and pricing.", "Watch spectrum auctions and online-safety legislation."),
        "Economic": ("Advertising follows the cycle; telecom cash flows are steadier but rate-sensitive.", "Track ad-market growth and telecom pricing discipline."),
        "Social": ("Streaming habits, social-media usage, and cord-cutting redistribute revenue.", "Watch engagement metrics and subscriber churn."),
        "Technological": ("AI-driven content, 5G/fibre build-outs, and platform competition set the pace.", "Monitor network capex and AI-feature adoption."),
        "Environmental": ("Network energy use and device lifecycles are emerging disclosure themes.", "Watch energy-efficiency targets for networks."),
        "Legal": ("Antitrust, privacy, and content-liability rulings can change business models.", "Recheck platform-liability cases and privacy enforcement."),
    },
    "Real Estate": {
        "Political": ("Housing policy, zoning, and immigration levels drive occupancy and rents.", "Watch rent-control proposals and housing-supply programs."),
        "Economic": ("Cap rates and financing costs follow interest rates; occupancy follows employment.", "Track rate moves, refinancing walls, and leasing spreads."),
        "Social": ("Hybrid work, urban migration, and household formation reshape demand by property type.", "Watch office utilization and rental-demand indicators."),
        "Technological": ("Data-centre and logistics demand grow while proptech changes operations.", "Monitor data-centre leasing and e-commerce logistics demand."),
        "Environmental": ("Energy-efficiency rules and climate exposure affect valuations and insurance.", "Watch building-emission standards and insurance premiums."),
        "Legal": ("Tenant law, tax changes for REITs, and lease disputes are recurring risks.", "Recheck REIT tax rules and landlord-tenant legislation."),
    },
    "Basic Materials": {
        "Political": ("Mining permits, resource nationalism, and trade policy shape supply.", "Watch permitting reforms and export restrictions."),
        "Economic": ("Commodity cycles, China demand, and the US dollar drive pricing.", "Track metal prices, construction activity, and inventory levels."),
        "Social": ("Community consent and labour relations affect project timelines.", "Watch strike risk and community agreements."),
        "Technological": ("Electrification demand for copper and lithium and process automation change economics.", "Monitor battery-chemistry shifts and mine automation."),
        "Environmental": ("Tailings, water use, and emissions rules are central operating risks.", "Watch environmental permits and tailings-safety audits."),
        "Legal": ("Royalty regimes, environmental litigation, and sanctions can alter cash flows.", "Recheck royalty legislation and pending environmental cases."),
    },
}

GENERIC_PESTEL: dict[str, tuple[str, str]] = {
    "Political": ("Regulation, tariffs, tax policy, and government exposure affect the operating backdrop.", "Watch policy announcements and sector-specific regulatory decisions."),
    "Economic": ("Rates, inflation, FX, demand, and valuation multiples influence the current setup.", "Stress-test the thesis against rate, currency, and demand changes."),
    "Social": ("Customer adoption, demographics, affordability, and trust shape demand.", "Watch customer behaviour, affordability, and preference changes."),
    "Technological": ("Technology adoption and disruption can change margins, capital needs, and competitive position.", "Monitor product launches, automation, platform shifts, and competitor execution."),
    "Environmental": ("Energy, climate, transition, and weather exposure may affect operating costs and demand.", "Watch transition costs, climate policy, and weather-sensitive risks."),
    "Legal": ("Litigation, competition rules, disclosure requirements, and sector-specific law can invalidate a setup.", "Recheck filings, investigations, and legal decisions before execution."),
}


def canonical_sector(value: Any) -> str | None:
    """Map free-form sector text onto one of the known PESTEL sector keys."""

    text = str(value or "").strip().lower()
    if not text:
        return None
    for sector in SECTOR_PESTEL:
        if sector.lower() == text:
            return sector
    aliases = {
        "financials": "Financial Services", "financial": "Financial Services", "banks": "Financial Services",
        "tech": "Technology", "information technology": "Technology", "software": "Technology", "semiconductors": "Technology",
        "health care": "Healthcare", "pharma": "Healthcare", "biotech": "Healthcare",
        "oil & gas": "Energy", "oil and gas": "Energy", "pipelines": "Energy",
        "consumer staples": "Consumer Defensive", "staples": "Consumer Defensive",
        "consumer discretionary": "Consumer Cyclical", "retail": "Consumer Cyclical",
        "telecom": "Communication Services", "media": "Communication Services", "communications": "Communication Services",
        "reit": "Real Estate", "reits": "Real Estate", "property": "Real Estate",
        "materials": "Basic Materials", "mining": "Basic Materials", "gold": "Basic Materials",
        "utility": "Utilities",
    }
    for alias, sector in aliases.items():
        if alias in text:
            return sector
    for sector in SECTOR_PESTEL:
        if sector.lower().split()[0] in text:
            return sector
    return None


def sector_pestel(sector: Any, market: str | None = None) -> list[dict[str, str]]:
    """Return the six PESTEL rows for a sector (generic rows when unknown)."""

    key = canonical_sector(sector)
    table = SECTOR_PESTEL.get(key or "", GENERIC_PESTEL)
    label = key or str(sector or "the selected sector")
    market_note = f" ({market})" if market else ""
    rows: list[dict[str, str]] = []
    for factor in PESTEL_FACTORS:
        current, forward = table.get(factor, GENERIC_PESTEL[factor])
        rows.append({
            "factor": factor,
            "sector": label,
            "current": current if key else f"{current} Sector: {label}{market_note}.",
            "forwardWatch": forward,
        })
    return rows


def profile_pestel(sectors: list[str] | None) -> list[dict[str, str]]:
    """Blend PESTEL reads across the sectors a plan touches."""

    canonical = [canonical_sector(item) for item in (sectors or [])]
    keys = [item for item in dict.fromkeys(canonical) if item]
    if not keys:
        return [{"factor": factor, "note": f"{current} {forward}"} for factor, (current, forward) in GENERIC_PESTEL.items()]
    rows: list[dict[str, str]] = []
    for factor in PESTEL_FACTORS:
        notes = [f"{sector}: {SECTOR_PESTEL[sector][factor][0]}" for sector in keys[:3]]
        watch = "; ".join(SECTOR_PESTEL[sector][factor][1] for sector in keys[:2])
        rows.append({"factor": factor, "note": " ".join(notes), "forwardWatch": watch, "sectors": ", ".join(keys[:3])})
    return rows
