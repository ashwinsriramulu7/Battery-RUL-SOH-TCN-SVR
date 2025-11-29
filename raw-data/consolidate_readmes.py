import os

OUTPUT_FILE = "merged_markdown.md"

def merge_markdown():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk("."):
            for filename in files:
                # Match any .md file (case-insensitive)
                if filename.lower().endswith(".txt"):
                    filepath = os.path.join(root, filename)

                    # Skip the output file itself
                    if os.path.abspath(filepath) == os.path.abspath(OUTPUT_FILE):
                        continue

                    print(f"Appending: {filepath}")

                    # Optional separator in the text file
                    outfile.write(f"\n\n# --- {filepath} ---\n\n")

                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())

    print(f"\nDone! Combined markdown saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_markdown()

