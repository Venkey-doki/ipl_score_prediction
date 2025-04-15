with open("C:\\Users\\swast\\Downloads\\archive\\ipl_score_prediction\\main.py", "rb") as f:
    # Read the entire file content
    data = f.read()

clean_data = data.replace(b"\x00", b"")

with open("main.py", "wb") as f:
    f.write(clean_data)
