"""Generate realistic German/Austrian OCR-style training samples with BIO labels.

Models the actual layout/wording of real Austrian documents (ÖGK
Arbeitsunfähigkeitsmeldung, Billa/Spar Kassenbon, Rechnung nach §11 UStG,
Lieferschein, Stammdatenblatt) and injects OCR-style noise (blank lines,
spacing jitter, light character corruption on non-entity tokens, and dates
written with spaces like "28. 12. 2026").

Each sample: {"document_type": str, "words": list[str], "tags": list[str]}
with BIO ``tags`` aligned 1:1 with whitespace-split ``words``.
"""

from __future__ import annotations

import random
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Value pools (Austrian flavoured)
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Hans", "Maria", "Thomas", "Anna", "Stefan", "Julia", "Michael", "Sandra",
    "Andreas", "Eva", "Markus", "Sabine", "Florian", "Christina", "Manuel",
    "Katharina", "Lukas", "Barbara", "Daniel", "Elisabeth", "Patrick", "Petra",
    "Johannes", "Claudia", "Bernhard", "Spongebob", "Gerald", "Nina", "Felix",
]
_LAST_NAMES = [
    "Maier", "Huber", "Berger", "Wagner", "Gruber", "Steinkellner", "Grubmüller",
    "Hofer", "Leitner", "Brunner", "Wimmer", "Moser", "Mayr", "Pichler",
    "Schwammkopf", "Eder", "Fuchs", "Aigner", "Reiter", "Schmid", "Lang",
    "Baumgartner", "Schwarz", "Winkler", "Aydogan", "Steiner", "Koch", "Bauer",
]

_COMPANY_CORE = [
    "Billa", "Spar", "Hofer", "REWE International", "Müller", "Bipa", "Interspar",
    "Metro", "Pagro", "Libro", "Thalia", "Hartlauer", "Kastner & Öhler",
    "Elektro Köck", "Bauhaus", "Obi", "Hornbach", "Lutz", "Kika", "Möbelix",
    "Apotheke am Ring", "Magenta Telekom", "A1 Telekom Austria", "Drei",
    "Gebrüder Weiss", "DPD Austria", "Post AG", "Würth Handel", "Strabag",
]
_LEGAL_FORMS = ["AG", "GmbH", "KG", "OG", "GesmbH", "e.U.", "GmbH & Co KG"]

_STREETS = [
    "Franz-Josefs-Kai", "Landstraße", "Getreidegasse", "Herrengasse",
    "Maria-Theresien-Straße", "Mariahilfer Straße", "Hauptstraße", "Bahnhofstraße",
    "Kärntner Straße", "Ringstraße", "Linzer Straße", "Wienerstraße",
    "Industriestraße", "Gewerbepark", "Dorfstraße", "Schulgasse", "Gartenweg",
]
_CITIES = [
    ("1010", "Wien"), ("1100", "Wien"), ("4020", "Linz"), ("5020", "Salzburg"),
    ("8010", "Graz"), ("6020", "Innsbruck"), ("9020", "Klagenfurt"),
    ("3100", "St. Pölten"), ("2700", "Wiener Neustadt"), ("4600", "Wels"),
    ("3531", "Niedernondorf"), ("6800", "Feldkirch"), ("8045", "Graz"),
]

_DIAGNOSES_REAL = [
    "akute Bronchitis", "Rückenschmerzen", "Migräne", "Influenza",
    "Bandscheibenvorfall", "grippaler Infekt", "Magen-Darm-Infekt",
    "Lumbago", "Gastritis", "Angina", "Verstauchung", "Sinusitis",
]
_DIAGNOSES_AU = ["Krankheit", "Unfall", "Krankheit/Unfall", "Berufskrankheit"]

_BICS = [
    "RLNWATWWXXX", "GIBAATWWXXX", "BKAUATWW", "SPGRAT2GXXX", "OPSKATWW",
    "VBOEATWW", "RZTIAT22", "EHBBAT2E", "BAWAATWW",
]
_BANKS = [
    "Raiffeisenbank", "Erste Bank", "BAWAG P.S.K.", "Bank Austria", "Sparkasse",
    "Volksbank", "Oberbank", "Hypo Tirol Bank", "Raiffeisenlandesbank",
]
_PAYMENTS = [
    "Bar", "MasterCard", "Visa", "Maestro", "Bankomat", "Kreditkarte",
    "EC-Karte", "Überweisung", "PayLife",
]
_VAT_RATES = ["20%", "10%", "13%", "0%"]
_INSURERS = ["ÖGK", "SVS", "BVAEB", "KFA", "ÖGK Wien", "ÖGK Steiermark", "ÖGK Tirol"]
_TIMES = ["08:15", "09:30", "11:45", "12:20", "14:05", "16:40", "17:55"]


def _rand_date() -> str:
    d, m = random.randint(1, 28), random.randint(1, 12)
    y = random.randint(2018, 2026)
    sep = random.choice([".", ".", "/", "-"])
    yy = str(y) if random.random() < 0.7 else f"{y % 100:02d}"
    base = f"{d:02d}{sep}{m:02d}{sep}{yy}"
    if random.random() < 0.25:  # OCR spaced date: "28. 12. 2026"
        base = f"{d:02d}{sep} {m:02d}{sep} {yy}"
    return base


def _rand_amount(big: bool = False) -> str:
    if big and random.random() < 0.5:
        whole = random.randint(1, 9)
        rest = random.randint(0, 999)
        cents = random.randint(0, 99)
        sep = random.choice([".", ""])
        return f"{whole}{sep}{rest:03d},{cents:02d}" if sep else f"{whole*1000+rest},{cents:02d}"
    whole = random.randint(1, 999)
    cents = random.randint(0, 99)
    return f"{whole},{cents:02d}" if random.random() < 0.6 else f"{whole}.{cents:02d}"


def _person() -> str:
    return f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"


def _doctor() -> str:
    title = random.choice(["Dr.", "Dr. med.", "Univ.-Prof. Dr.", "Dr."])
    name = f"{title} {random.choice(_LAST_NAMES)}"
    if random.random() < 0.35:
        name += f" & Dr. {random.choice(_LAST_NAMES)}"
    return name


def _company() -> str:
    core = random.choice(_COMPANY_CORE)
    return f"{core} {random.choice(_LEGAL_FORMS)}" if random.random() < 0.7 else core


def _address() -> str:
    plz, city = random.choice(_CITIES)
    return f"{plz} {city} {random.choice(_STREETS)} {random.randint(1, 199)}"


def _svnr() -> str:
    return f"{random.randint(1000,9999)} {random.randint(10,31):02d} {random.randint(1,12):02d} {random.randint(10,99)}"


def _vat_id() -> str:
    return f"ATU{random.randint(10000000, 99999999)}"


def _iban() -> str:
    return "AT" + "".join(str(random.randint(0, 9)) for _ in range(18))


def _phone() -> str:
    fmts = ["+43 {a} {b}", "0{a}/{b}", "0{a} {b}", "+43-{a}-{b}"]
    return random.choice(fmts).format(a=random.randint(1, 699), b=random.randint(100000, 9999999))


def _email(company: str) -> str:
    base = company.split()[0].lower()
    for a, b in (("ö", "oe"), ("ä", "ae"), ("ü", "ue"), ("ß", "ss"), ("&", "")):
        base = base.replace(a, b)
    base = "".join(c for c in base if c.isalnum())
    return f"{random.choice(['office', 'kontakt', 'info', 'buchhaltung'])}@{base}.at"


def _website(company: str) -> str:
    base = company.split()[0].lower()
    for a, b in (("ö", "oe"), ("ä", "ae"), ("ü", "ue"), ("ß", "ss"), ("&", "")):
        base = base.replace(a, b)
    base = "".join(c for c in base if c.isalnum())
    return f"www.{base}.at"


def _bic() -> str:
    return random.choice(_BICS)


def _bank() -> str:
    return random.choice(_BANKS)


def _customer_number() -> str:
    return random.choice([
        f"K-{random.randint(1000, 99999)}",
        f"KD{random.randint(10000, 999999)}",
        str(random.randint(100000, 999999)),
    ])


def _order_number() -> str:
    return random.choice([
        f"AB-{random.randint(10000, 99999)}",
        f"BST-{random.randint(2020, 2026)}-{random.randint(100, 999)}",
        f"45{random.randint(10000000, 99999999)}",
    ])


def _company_register() -> str:
    return f"FN {random.randint(100000, 999999)}{random.choice('abdefghikmpstvwxyz')}"


# ---------------------------------------------------------------------------
# Annotation + OCR noise
# ---------------------------------------------------------------------------

def _annotate(words: list[str], spans: list[tuple[str, str]]) -> list[str]:
    """Assign BIO tags. ``spans`` ordered (field, phrase); each phrase matched
    to the first not-yet-consumed word subsequence."""
    tags = ["O"] * len(words)
    consumed = [False] * len(words)
    for field, phrase in spans:
        pw = phrase.split()
        n = len(pw)
        if n == 0:
            continue
        for i in range(len(words) - n + 1):
            if any(consumed[i : i + n]):
                continue
            if all(words[i + j] == pw[j] for j in range(n)):
                tags[i] = f"B-{field}"
                consumed[i] = True
                for j in range(1, n):
                    tags[i + j] = f"I-{field}"
                    consumed[i + j] = True
                break
    return tags


_OCR_SWAP = {"O": "0", "0": "O", "l": "1", "I": "1", "S": "5", "B": "8"}


def _ocr_noise(lines: list[str], protect: set[str]) -> str:
    out: list[str] = []
    for line in lines:
        if line == "":
            out.append("")
            continue
        toks = []
        for tok in line.split():
            if tok not in protect and random.random() < 0.04 and len(tok) > 2:
                idx = random.randrange(len(tok))
                ch = tok[idx]
                if ch in _OCR_SWAP and random.random() < 0.5:
                    tok = tok[:idx] + _OCR_SWAP[ch] + tok[idx + 1 :]
            toks.append(tok)
        out.append(" ".join(toks))
        if random.random() < 0.25:
            out.append("")
    return "\n".join(out)


def _finalize(lines: list[str], spans: list[tuple[str, str]], doc_type: str) -> dict[str, Any]:
    protect: set[str] = set()
    for _, phrase in spans:
        protect.update(phrase.split())
    text = _ocr_noise(lines, protect)
    words = text.split()
    return {"document_type": doc_type, "words": words, "tags": _annotate(words, spans)}


# ---------------------------------------------------------------------------
# Document templates
# ---------------------------------------------------------------------------

def _sample_invoice() -> dict[str, Any]:
    company, address, vat_id = _company(), _address(), _vat_id()
    phone, email = _phone(), _email(company)
    customer = _company() if random.random() < 0.5 else _person()
    cust_addr = _address()
    cust_no, order_no = _customer_number(), _order_number()
    inv_no = random.choice([f"RE-{random.randint(10000,99999)}",
                            f"{random.randint(2018,2026)}/{random.randint(1,9999):04d}",
                            f"RG{random.randint(100000,999999)}"])
    date, deliv, due = _rand_date(), _rand_date(), _rand_date()
    item_price = _rand_amount(big=True)
    subtotal, vat, total = _rand_amount(big=True), _rand_amount(), _rand_amount(big=True)
    rate = random.choice(_VAT_RATES)
    iban, bic = _iban(), _bic()

    lines = [
        "RECHNUNG", company, address, f"UID: {vat_id}",
        f"Tel: {phone}", f"E-Mail: {email}",
        "Rechnungsempfänger:", customer, cust_addr,
        f"Kundennummer: {cust_no}",
        f"Rechnungsnummer: {inv_no}",
        f"Rechnungsdatum: {date}",
        f"Lieferdatum: {deliv}",
        f"Bestellnummer: {order_no}",
        "Pos Bezeichnung Menge Einzelpreis",
        f"1 Dienstleistung 1 {item_price}",
        f"Nettobetrag: {subtotal} EUR",
        f"Steuersatz: {rate}",
        f"MwSt-Betrag: {vat} EUR",
        f"Gesamtbetrag: {total} EUR",
        f"Zahlbar bis {due}",
        f"Bankverbindung IBAN {iban} BIC {bic}",
    ]
    spans = [
        ("company", company), ("address", address), ("vat_id", vat_id),
        ("phone", phone), ("email", email),
        ("customer", customer), ("customer_address", cust_addr),
        ("customer_number", cust_no), ("invoice_number", inv_no),
        ("date", date), ("delivery_date", deliv), ("order_number", order_no),
        ("subtotal", subtotal), ("vat_rate", rate), ("vat", vat),
        ("total", total), ("currency", "EUR"), ("due_date", due),
        ("iban", iban), ("bic", bic),
    ]
    return _finalize(lines, spans, "invoice")


def _sample_receipt() -> dict[str, Any]:
    company, address, vat_id = _company(), _address(), _vat_id()
    phone = _phone()
    date, time = _rand_date(), random.choice(_TIMES)
    total, vat, subtotal = _rand_amount(big=True), _rand_amount(), _rand_amount(big=True)
    rate = random.choice(["10%", "20%", "13%"])
    beleg = str(random.randint(100000, 999999))
    payment = random.choice(_PAYMENTS)

    items = []
    for _ in range(random.randint(2, 4)):
        items += [random.choice(["Cl. ESL-Vollmilch 1L", "Lindt Excellence",
                                  "Clever Schokolade", "Mars Classic 4er",
                                  "Semmel 5 Stk", "Butter 250g"]),
                  random.choice(["B", "A", ""]), _rand_amount()]

    lines = [
        company, address, f"TEL: {phone}", f"ATU {vat_id[3:]}",
        f"Datum: {date}", f"Zeit: {time}", *items,
        "Summe", "EUR", total, payment, "BEZAHLT",
        f"Beleg Nr. :{beleg}",
        f"B: {rate} MwSt von {subtotal} = {vat}",
    ]
    spans = [
        ("company", company), ("address", address), ("vat_id", vat_id),
        ("phone", phone), ("date", date), ("time", time),
        ("total", total), ("currency", "EUR"), ("payment_method", payment),
        ("receipt_number", beleg), ("vat_rate", rate),
        ("subtotal", subtotal), ("vat", vat),
    ]
    return _finalize(lines, spans, "receipt")


def _sample_doctor_note(doc_type: str = "doctor_note") -> dict[str, Any]:
    patient, doctor = _person(), _doctor()
    birth, start, end, issue = _rand_date(), _rand_date(), _rand_date(), _rand_date()
    insurer, svnr = random.choice(_INSURERS), _svnr()
    p_addr, d_addr = _address(), _address()
    practice_no = str(random.randint(100000, 999999))

    if doc_type == "doctor_note" and random.random() < 0.65:
        diagnosis = random.choice(_DIAGNOSES_AU)
        lines = [
            random.choice(["Arbeitsunfähigkeitsmeldung", "Arbeitsfähigkeitsmeldung"]),
            f"Versicherungsträger: {insurer}",
            "Familienname, Vorname(n):", patient,
            f"Geburtsdatum: {birth}",
            "Versicherungsnummer:", svnr,
            "Krankenstandsadresse:", p_addr,
            f"Arbeitsunfähig von: {start}",
            "Letzter Tag der", f"Arbeitsunfähigkeit: {end}",
            doctor, practice_no, d_addr,
            "Grund der Arbeitsunfähigkeit:",
            "Anstaltspflege:", diagnosis, "der Arbeitsunfähigkeit:",
            f"Ausstellungsdatum {issue}",
            "Unterschrift und Stempel der Ärztin/des Arztes",
        ]
    else:
        diagnosis = random.choice(_DIAGNOSES_REAL)
        lines = [
            "ÄRZTLICHE BESTÄTIGUNG", doctor, d_addr,
            f"Versicherungsträger: {insurer}",
            f"Patient/in: {patient}",
            f"Geburtsdatum: {birth}",
            f"Versicherungsnummer: {svnr}",
            f"Adresse: {p_addr}",
            f"Diagnose: {diagnosis}",
            f"Arbeitsunfähig von {start} bis {end}",
            f"Ausstellungsdatum: {issue}",
            "Unterschrift und Stempel",
        ]
    spans = [
        ("patient", patient), ("birth_date", birth), ("insurer", insurer),
        ("insurance_number", svnr), ("address", p_addr),
        ("doctor", doctor), ("doctor_address", d_addr),
        ("start_date", start), ("end_date", end), ("issue_date", issue),
        ("diagnosis", diagnosis),
    ]
    return _finalize(lines, spans, doc_type)


def _sample_care_leave() -> dict[str, Any]:
    patient, doctor = _person(), _doctor()
    birth, start, end, issue = _rand_date(), _rand_date(), _rand_date(), _rand_date()
    insurer, svnr = random.choice(_INSURERS), _svnr()
    p_addr, d_addr = _address(), _address()
    diagnosis = random.choice(_DIAGNOSES_REAL)

    lines = [
        "Ärztliche Bestätigung", "über die Notwendigkeit der Pflege",
        f"Versicherungsträger: {insurer}",
        doctor, d_addr,
        f"Erkrankte Person: {patient}",
        f"Geburtsdatum: {birth}",
        f"Versicherungsnummer: {svnr}",
        f"Wohnadresse: {p_addr}",
        f"Diagnose: {diagnosis}",
        f"Pflegefreistellung von {start} bis {end}",
        f"Ausstellungsdatum: {issue}",
        "Unterschrift und Stempel der Ärztin/des Arztes",
    ]
    spans = [
        ("patient", patient), ("birth_date", birth), ("insurer", insurer),
        ("insurance_number", svnr), ("address", p_addr),
        ("doctor", doctor), ("doctor_address", d_addr),
        ("diagnosis", diagnosis),
        ("start_date", start), ("end_date", end), ("issue_date", issue),
    ]
    return _finalize(lines, spans, "care_leave")


def _sample_delivery_note() -> dict[str, Any]:
    company, address, vat_id = _company(), _address(), _vat_id()
    customer = _company() if random.random() < 0.5 else _person()
    cust_addr, cust_no = _address(), _customer_number()
    date, deliv = _rand_date(), _rand_date()
    deliv_no = random.choice([f"LS-{random.randint(1000,99999)}",
                              f"LF{random.randint(100000,999999)}",
                              f"{random.randint(2020,2026)}-{random.randint(1,9999):04d}"])
    order_no = _order_number()

    lines = [
        "LIEFERSCHEIN", company, address, f"UID: {vat_id}",
        f"Lieferschein-Nr.: {deliv_no}",
        f"Ausstellungsdatum: {date}",
        f"Lieferdatum: {deliv}",
        f"Bestellnummer: {order_no}",
        "Empfänger:", customer, cust_addr,
        f"Kundennummer: {cust_no}",
        "Pos Artikel Menge", "1 Ware 10 Stk",
    ]
    spans = [
        ("company", company), ("address", address), ("vat_id", vat_id),
        ("delivery_number", deliv_no), ("date", date), ("delivery_date", deliv),
        ("order_number", order_no), ("customer", customer),
        ("customer_address", cust_addr), ("customer_number", cust_no),
    ]
    return _finalize(lines, spans, "delivery_note")


def _sample_master_data() -> dict[str, Any]:
    company, address, vat_id = _company(), _address(), _vat_id()
    phone, fax = _phone(), _phone()
    email, website = _email(company), _website(company)
    contact = _person()
    iban, bic, bank = _iban(), _bic(), _bank()
    fn = _company_register()

    lines = [
        "STAMMDATEN", company, address,
        f"Ansprechpartner: {contact}",
        f"Tel: {phone}", f"Fax: {fax}",
        f"E-Mail: {email}", f"Web: {website}",
        f"Bank: {bank}", f"IBAN: {iban}", f"BIC: {bic}",
        f"UID: {vat_id}", f"Firmenbuchnummer: {fn}",
    ]
    spans = [
        ("company", company), ("address", address),
        ("contact_person", contact), ("phone", phone), ("fax", fax),
        ("email", email), ("website", website), ("bank_name", bank),
        ("iban", iban), ("bic", bic), ("vat_id", vat_id),
        ("company_register_number", fn),
    ]
    return _finalize(lines, spans, "master_data")


_GENERATORS: dict[str, Callable[[], dict[str, Any]]] = {
    "invoice": _sample_invoice,
    "receipt": _sample_receipt,
    "doctor_note": lambda: _sample_doctor_note("doctor_note"),
    "care_leave": _sample_care_leave,
    "delivery_note": _sample_delivery_note,
    "master_data": _sample_master_data,
}


def generate_synthetic_dataset(
    samples_per_type: int = 1200,
    document_types: list[str] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    random.seed(seed)
    types = document_types or list(_GENERATORS.keys())
    samples: list[dict[str, Any]] = []
    for doc_type in types:
        gen = _GENERATORS[doc_type]
        for _ in range(samples_per_type):
            samples.append(gen())
    random.shuffle(samples)
    return samples


if __name__ == "__main__":
    ds = generate_synthetic_dataset(samples_per_type=1)
    for s in ds:
        print("===", s["document_type"], "===")
        for w, t in zip(s["words"], s["tags"]):
            print(f"  {w}{'' if t == 'O' else '  <' + t + '>'}")
        print()
