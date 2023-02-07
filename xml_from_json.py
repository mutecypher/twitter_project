import json
import xml.etree.ElementTree as ET

# Load the JSON data from a file or any other source
with open("data.json", "r") as file:
    data = json.load(file)

# Build the XML tree from the JSON data
root = ET.Element("root")
for key, value in data.items():
    element = ET.SubElement(root, key)
    element.text = str(value)

# Write the XML tree to a file or any other destination
xml_data = ET.tostring(root, encoding="unicode")
with open("data.xml", "w") as file:
    file.write(xml_data)
