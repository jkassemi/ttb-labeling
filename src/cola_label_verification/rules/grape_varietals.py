import unicodedata
from functools import lru_cache
from typing import Final

from cola_label_verification.rules.common import (
    build_finding,
    is_imported,
    require_beverage_type,
)
from cola_label_verification.rules.models import Finding, FindingStatus, RuleContext

RULE_ID: Final[str] = "grape_varietals"
DEFAULT_FIELD: Final[str] = "grape_varietals"
APPELLATION_FIELD: Final[str] = "appellation_of_origin"

# Derived from TTB.gov sources; see context/grape-variety-designations.txt.
ADMINISTRATIVELY_APPROVED_VARIETALS: Final[tuple[str, ...]] = (
    "Albillo Mayor",
    "Alexander",
    "Ambulo Blanc",
    "Amigne",
    "Arandell",
    "Arinarnoa",
    "Aromella",
    "Arvine",
    "Assyrtiko",
    "Baga",
    "Bianchetta Trevigiana",
    "Black Spanish",
    "Bluebell",
    "Bourboulenc",
    "Brachetto",
    "By George",
    "Cabernet Dorsa",
    "Cabernet Volos",
    "Caladoc",
    "Caminante Blanc",
    "Camminare Noir",
    "Cannonau",
    "Caprettone",
    "Carignano",
    "Carricante",
    "Catarratto",
    "Chisago",
    "Ciliegiolo",
    "Cinsault",
    "Clarion",
    "Coda di Volpe",
    "Colorino",
    "Courbu Blanc",
    "Crimson Pearl",
    "Diana",
    "Dobricic",
    "Enchantment",
    "Errante Noir",
    "Esprit",
    "Falanghina",
    "Fleurtai",
    "Frappato",
    "Frontenac Blanc",
    "Garanoir",
    "Garnacha Roja",
    "Geneva Red",
    "Godello",
    "Golubok",
    "Gouais Blanc",
    "Greco",
    "Greco Bianco",
    "Grenache Gris",
    "Gros Manseng",
    "Humagne Rouge",
    "Itasca",
    "Jacquez",
    "Jupiter",
    "King of the North",
    "Koshu",
    "L'Acadie Blanc",
    "Lambrusca di Alessandria",
    "Lomanto",
    "Loureiro",
    "Macabeo",
    "Madeleine Angevine",
    "Madeleine Sylvaner",
    "Marquis",
    "Marselan",
    "Mavron",
    "Mencia",
    "Merlot Kanthus",
    "Moschofilero",
    "Mourtaou",
    "Muscardin",
    "Mustang",
    "Nerello Mascalese",
    "Opportunity",
    "Pallagrello Bianco",
    "Parellada",
    "Paseante Noir",
    "Pecorino",
    "Petite Pearl",
    "Picardan",
    "Pinot Bianco",
    "Pinot Nero",
    "Plymouth",
    "Poulsard",
    "Prieto Picudo",
    "Regent",
    "Ribolla Gialla",
    "Rieslaner",
    "Riverbank",
    "Roditis",
    "Rose of Peru",
    "Rossese",
    "Ruche",
    "San Marco",
    "Saperavi",
    "Sauvignon Kretos",
    "Sauvignon Rytos",
    "Savagnin",
    "Savagnin Blanc",
    "Schiava Grossa",
    "Schioppettino",
    "Schönburger",
    "Sheridan",
    "Somerset Seedless",
    "Soreli",
    "Southern Cross",
    "Terret Noir",
    "Thiakon",
    "Tibouren",
    "Tinta Amarela",
    "Tinta Cao",
    "Tinta Roriz",
    "Torrontés Riojano",
    "Touriga Nacional",
    "Treixadura",
    "Trentina",
    "Trincadeira",
    "Vaccarèse",
    "Valjohn",
    "Verdejo",
    "Verdicchio",
    "Vranac",
    "Xarel·lo",
    "Xynisteri",
)

VARIETAL_SEPARATORS: Final[tuple[str, ...]] = (" & ", " and ", "/", ",")


def normalize_grape_name(name: str) -> str:
    """Normalize grape varietal names for loose matching."""
    normalized = unicodedata.normalize("NFKD", name)
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return "".join(char for char in stripped.casefold() if char.isalnum())


def split_grape_varietals(value: str) -> list[str]:
    """Split a varietal string into candidate names."""
    normalized = value
    for sep in VARIETAL_SEPARATORS:
        normalized = normalized.replace(sep, "|")
    parts = [part.strip() for part in normalized.split("|")]
    return [part for part in parts if part]


@lru_cache(maxsize=1)
def load_administratively_approved_varieties() -> set[str]:
    """Load administratively approved varietal names."""
    return {normalize_grape_name(name) for name in ADMINISTRATIVELY_APPROVED_VARIETALS}


def grape_varietals(context: RuleContext) -> Finding:
    gate = require_beverage_type(
        context,
        allowed={"wine"},
        rule_id=RULE_ID,
        field=DEFAULT_FIELD,
    )
    if gate is not None:
        return gate

    label_info = context.label_info
    label_value = label_info.grape_varietals.value
    if not label_value:
        return build_finding(
            RULE_ID,
            "fail",
            "Grape varietals not detected on the label.",
            field=DEFAULT_FIELD,
        )

    appellation_present = bool(label_info.appellation_of_origin.value)
    missing_appellation = not appellation_present

    approval_status: FindingStatus = "pass"
    approval_message = "Grape varietal designation is on the approved list."
    unknown_varietals: list[str] = []

    imported = is_imported(context.application_fields)
    if imported is None:
        approval_status = "not_evaluated"
        approval_message = (
            "Source of product not provided; cannot validate varietal approval."
        )
    elif imported:
        approval_status = "not_applicable"
        approval_message = (
            "Imported wines are not restricted to the domestic varietal list."
        )
    else:
        approved = load_administratively_approved_varieties()
        if not approved:
            approval_status = "not_evaluated"
            approval_message = "Approved varietal list not available."
        else:
            for varietal in split_grape_varietals(label_value):
                normalized = normalize_grape_name(varietal)
                if normalized not in approved:
                    unknown_varietals.append(varietal)
            if unknown_varietals:
                approval_status = "needs_review"
                approval_message = (
                    "Grape varietal designation may not be approved for domestic wine."
                )

    evidence: dict[str, object] = {"appellation_present": appellation_present}
    if imported is not None:
        evidence["imported"] = imported
    if unknown_varietals:
        evidence["unknown_varietals"] = unknown_varietals
    evidence["approval_status"] = approval_status

    field = DEFAULT_FIELD
    if missing_appellation and approval_status != "needs_review":
        field = APPELLATION_FIELD

    if missing_appellation and approval_status == "needs_review":
        message = (
            "Appellation not detected with a grape varietal designation, and one or "
            "more varietals may not be approved for domestic wine."
        )
        status: FindingStatus = "needs_review"
    elif missing_appellation:
        message = "Appellation not detected with a grape varietal designation."
        status = "needs_review"
    elif approval_status == "needs_review":
        message = approval_message
        status = "needs_review"
    elif approval_status == "not_evaluated":
        message = approval_message
        status = "not_evaluated"
    else:
        message = approval_message
        status = "pass"

    severity = (
        "info" if status in {"pass", "not_applicable", "not_evaluated"} else "warning"
    )
    return build_finding(
        RULE_ID,
        status,
        message,
        field=field,
        severity=severity,
        evidence=evidence,
    )
