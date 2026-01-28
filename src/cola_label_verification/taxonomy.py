from typing import Final

# Derived from TTB.gov class/type guidance; see context/spirits-class-types.txt.
SPIRITS_CLASS_KEYWORDS: Final = (
    "vodka",
    "gin",
    "rum",
    "whiskey",
    "whisky",
    "bourbon",
    "rye",
    "tequila",
    "mezcal",
    "brandy",
    "liqueur",
    "cordial",
    "schnapps",
    "absinthe",
    "spirit",
    "liqueurs",
    "spirits",
)
MALT_CLASS_KEYWORDS: Final = (
    "malt beverage",
    "malt liquor",
    "beer",
    "ale",
    "lager",
    "lager beer",
    "porter",
    "stout",
    "near beer",
    "cereal beverage",
)
WINE_CLASS_KEYWORDS: Final = (
    "wine",
    "red wine",
    "white wine",
    "rose",
    "sparkling wine",
    "champagne",
    "dessert wine",
    "table wine",
    "fortified wine",
    "port",
    "sherry",
    "vermouth",
    "sake",
)
CLASS_KEYWORDS: Final = (
    SPIRITS_CLASS_KEYWORDS + MALT_CLASS_KEYWORDS + WINE_CLASS_KEYWORDS
)
