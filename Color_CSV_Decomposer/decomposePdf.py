import tabula

def extract_tables_from_pdf(pdf_path):
    # Use tabula to read tables in the PDF into a list of DataFrames
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=False)

    # Display the extracted tables
    for idx, table in enumerate(tables, 1):
        print(f"Table {idx}:")
        print(table)
        print("-" * 50)

    return tables

pdf_path = "ColorGuide1.pdf"
tables = extract_tables_from_pdf(pdf_path)

print(tables)
