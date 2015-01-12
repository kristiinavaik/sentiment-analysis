from xml.etree import ElementTree as ET


def parse(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()

    results = []

    for sentence in root:
        text = sentence.find('text').text
        aspect_terms = sentence.find('aspectTerms')
        if not aspect_terms:
            continue
        polarity = aspect_terms[0].attrib.get('polarity')
        results.append("%s\t%s" % (polarity, text))
    return results


def write_results(file_name, lines):
    with open(file_name, 'w') as f: 
        for line in lines:
            f.write("%s\n" % line)

laptops_results = parse('Laptops_Test_Gold.xml')
restaraunts_results = parse('Restaurants_Test_Gold.xml')

write_results('laptops.txt', laptops_results)
write_results('restaraunts.txt', restaraunts_results)
