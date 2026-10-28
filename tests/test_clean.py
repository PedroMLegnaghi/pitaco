import pytest

from pitaco.data.clean import clean_text, normalize_whitespace, scrub_pii


def test_whitespace_and_line_breaks_collapse():
    assert normalize_whitespace("  Faltou item\r\nMe ligue  ") == "Faltou item Me ligue"


@pytest.mark.parametrize(
    ("text", "expected", "kind"),
    [
        ("Faltou item. Me ligue 34 99978 4951", "Faltou item. Me ligue [TELEFONE]", "phone"),
        ("ligar (11) 98765-4321 urgente", "ligar [TELEFONE] urgente", "phone"),
        ("meu email é ana.silva@gmail.com", "meu email é [EMAIL]", "email"),
        ("comprei em www.lannister.com ok", "comprei em [URL] ok", "url"),
        ("CPF 123.456.789-09 errado", "CPF [CPF] errado", "cpf"),
        ("CNPJ 12.345.678/0001-90", "CNPJ [CNPJ]", "cnpj"),
        ("entregar no CEP 90040-060", "entregar no CEP [CEP]", "cep"),
        ("O Sr. Bruno me atendeu bem", "O Sr. [NOME] me atendeu bem", "name_honorific"),
        ("pós venda através do Sr Renato", "pós venda através do Sr [NOME]", "name_honorific"),
        ("Tudo certo! Att   Elenice.", "Tudo certo! Att [NOME]", "name_signoff"),
        (
            "aguardo retorno urgente att alice bretanha",
            "aguardo retorno urgente att [NOME]",
            "name_signoff",
        ),
        (
            "Olá, meu nome é Carlos e não recebi",
            "Olá, meu nome é [NOME] e não recebi",
            "name_intro",
        ),
        ("me chamo João Silva, quero trocar", "me chamo [NOME], quero trocar", "name_intro"),
    ],
)
def test_pii_is_replaced(text, expected, kind):
    scrubbed, hits = scrub_pii(text)
    assert scrubbed == expected
    assert hits[kind] >= 1


@pytest.mark.parametrize(
    "text",
    [
        "Ainda não usei. .mas parece Sr muito boa e é muito bonita.",  # "Sr" typo for "ser"
        "Comprei e amei, chegou antes do prazo",  # capitalized verbs are not names
        "Produto custou R$ 89,90 e chegou em 3 dias",  # prices and small numbers stay
        "Loja lannister cumpre o prazo",  # Olist's own pseudonym for the seller
    ],
)
def test_ordinary_text_is_untouched(text):
    scrubbed, hits = scrub_pii(text)
    assert scrubbed == text
    assert not hits


def test_clean_text_reports_counts():
    text, hits = clean_text("Me ligue\r\n34 99978 4951 ou 34 98888 7777")
    assert text == "Me ligue [TELEFONE] ou [TELEFONE]"
    assert hits["phone"] == 2
