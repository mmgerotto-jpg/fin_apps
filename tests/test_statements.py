import io
from pathlib import Path

from optionsapp.statements import parse_csv, parse_statement

SAMPLE = Path(__file__).parent.parent / "samples" / "sample_statement.csv"


def test_fidelity_style_csv():
    hs = {h.symbol: h for h in parse_csv(SAMPLE.read_text())}
    assert set(hs) == {"AAPL", "MSFT", "KO"}  # money market & option rows excluded
    assert hs["AAPL"].quantity == 300
    assert hs["MSFT"].last_price == 425.5


def test_schwab_style_with_preamble_and_semicolons():
    text = 'Positions for account ...\n\n"Symbol";"Description";"Qty (Quantity)";"Price"\n"BRK.B";"Berkshire";"10";"$450.00"\n"T";"AT&T";"1,200";"$20.00"\n"Account Total";"";"";""\n'
    hs = {h.symbol: h.quantity for h in parse_csv(text)}
    assert hs == {"BRK-B": 10, "T": 1200}


def test_duplicate_lots_are_summed():
    text = "Ticker,Shares\nAAPL,100\nAAPL,50\n"
    assert parse_csv(text)[0].quantity == 150


def _make_pdf(lines):
    """Minimal PDF with text lines, no external deps."""
    content = "BT /F1 10 Tf 40 760 Td 14 TL " + " ".join(f"({l}) Tj T*" for l in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"]
    out, offs = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, o in enumerate(objs, 1):
        offs.append(out.tell())
        out.write(f"{i} 0 obj\n{o}\nendobj\n".encode())
    x = out.tell()
    out.write(f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode())
    for o in offs:
        out.write(f"{o:010d} 00000 n \n".encode())
    out.write(f"trailer << /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF".encode())
    return out.getvalue()


def test_pdf_text_fallback():
    pdf = _make_pdf(["Brokerage Statement  Page 1", "Symbol Description Quantity Price Value",
                     "AAPL Apple Inc 300 230.10 69,030.00", "KO Coca-Cola Co 250 68.20 17,050.00",
                     "Total 86,080.00"])
    hs = {h.symbol: h.quantity for h in parse_statement("stmt.pdf", pdf)}
    assert hs == {"AAPL": 300, "KO": 250}
