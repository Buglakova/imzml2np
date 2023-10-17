import re
from collections import Counter

CLEAN_REGEXP = re.compile(r"[.=]")
FORMULA_REGEXP = re.compile(r"(\[13C\]|[A-Z][a-z]*)([0-9]*)")
ADDUCT_VALIDATE_REGEXP = re.compile(r"^([+-]([A-Z][a-z]*[0-9]*)+)+$")
ADDUCT_REGEXP = re.compile(r"([+-])([A-Za-z0-9]+)")


def parse_formula(formula):
    return [(elem, int(n or "1")) for (elem, n) in FORMULA_REGEXP.findall(formula)]


def _chnops_13c_sort(ion_elements: Counter):

    """
    Reorder elements to be consistently ordered per the method in pyMSpec.
    Order: [13C], then "CHNOPS", then any other elements in alphabetic order.

    :param ion_elements:
        Counter of elements in formula.
    :return:
        String with ordered elements.
    """
    keys = list(ion_elements.keys())
    if "[13C]" in keys:
        result = ["[13C]"]
        keys.remove("[13C]")
    else:
        result = []
    return result + [
        *"CHNOPS",
        *sorted(key for key in keys if len(key) > 1 or key not in "CHNOPS"),
    ]


def format_modifiers(*adducts):
    return "".join(
        adduct for adduct in adducts if adduct and adduct not in ("[M]+", "[M]-")
    )


def format_charge(charge):
    if not charge:
        return ""
    if charge == 1:
        return "+"
    if charge == -1:
        return "-"
    return format(int(charge), "+0")


def format_ion_formula(formula, *adducts, charge=None):
    return formula + format_modifiers(*adducts) + format_charge(charge)


def _format_formula(elements: Counter):
    """_summary_

    :param elements:
        Counter, keys are elements, counts are number of atoms.
    :return:
        String for ion formula with reordered elements.
    """
    element_order = _chnops_13c_sort(elements)

    ion_formula_parts = []
    for elem in element_order:
        count = elements[elem]
        if count != 0:
            ion_formula_parts.append(elem)
            if count > 1:
                ion_formula_parts.append(str(count))

    return "".join(ion_formula_parts)


def generate_ion_formula(formula: str, *adducts: str):
    """
    Calculate the resulting molecular formula after applying a set of transformations,
    e.g. `generate_ion_formula('H2O', '+H', '-O')` => `'H3'`
    Throw an error if any component isn't formatted correctly,
    or if any step of the transformation sequence would
    create an impossible molecule with no elements, or a negative quantity of any element.

    :param formula:
        Molecular formula.
    :param adducts:
        Transformations to apply to the formula. Should start with + or -.
    :return:
        Molecular formula for ion taking into account adducts.
    """
    formula = CLEAN_REGEXP.sub("", formula)
    adducts = [adduct for adduct in adducts if adduct]

    ion_elements = Counter(dict(parse_formula(formula)))

    for adduct in adducts:
        if adduct in ("[M]+", "[M]-"):
            continue
        if not ADDUCT_VALIDATE_REGEXP.match(adduct):
            raise ParseFormulaError(f"Invalid adduct: {adduct}")

        for operation, adduct_part in ADDUCT_REGEXP.findall(adduct):
            assert operation in ("+", "-"), "Adduct should be prefixed with + or -"
            for elem, n in parse_formula(adduct_part):
                if operation == "+":
                    ion_elements[elem] += n
                else:
                    ion_elements[elem] -= n
                    if ion_elements[elem] < 0:
                        raise ParseFormulaError(
                            f"Negative total element count for {elem}"
                        )

    if not any(count > 0 for count in ion_elements.values()):
        raise ParseFormulaError("No remaining elements")

    return _format_formula(ion_elements)


def safe_generate_ion_formula(*parts):
    """
    Wrapper for generate_ion_formula to catch exceptions.
    """
    try:
        return generate_ion_formula(*(part for part in parts if part))
    except ParseFormulaError:
        return None
