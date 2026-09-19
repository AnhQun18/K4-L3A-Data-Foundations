import os
import urllib.request
from urllib.parse import urlparse
import datetime
from pathlib import Path
try:
    import pymupdf4llm
except ImportError:
    print("Please install pymupdf4llm: pip install pymupdf4llm")
    exit(1)

def extract_pdf_text(pdf_path):
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        return md_text
    except Exception as e:
        print(f"Error converting PDF to MD: {e}")
        return ""

def main():
    links_file = "Links.txt"
    out_dir = Path("data/neu-quy-dinh")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(links_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
        
    for url in urls:
        if not url.lower().endswith(".pdf"):
            continue
            
        print(f"Processing PDF: {url}")
        filename = os.path.basename(urlparse(url).path)
        doc_id = filename.replace(".pdf", "").replace("%20", "-")
        # cleanup doc_id
        doc_id = "".join(c if c.isalnum() else "-" for c in doc_id).strip("-").lower()
        
        pdf_path = out_dir / filename
        
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0',
                'Cookie': 'D1N=d8d4a6ef568620ee6c87c9d7803545b4'
            })
            with urllib.request.urlopen(req) as response, open(pdf_path, 'wb') as out_file:
                out_file.write(response.read())
                
            text = extract_pdf_text(str(pdf_path))
            
            # create md
            md_path = out_dir / f"{doc_id}.md"
            frontmatter = f"""---
doc_id: {doc_id}
title: "{doc_id.replace('-', ' ').title()}"
source_url: "{url}"
retrieved_at: "{datetime.date.today().isoformat()}"
document_version: "not-stated"
audience: "student"
department: "quan-ly-dao-tao"
category: "quy-dinh"
---

# {doc_id.replace('-', ' ').title()}

{text}
"""
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(frontmatter)
            print(f"Saved to {md_path}")
            
            # remove pdf to keep clean
            os.remove(pdf_path)
            
        except Exception as e:
            print(f"Failed to process {url}: {e}")

if __name__ == "__main__":
    main()
