import xml.etree.ElementTree as ET


def getXMLLabel(xmlFile):
    # create element tree object
    tree = ET.parse(xmlFile)
    return tree

# xmlTree = getXMLLabel("./Annotations/red1.eaf")

def getFrameLabel(xmlTree, frame):
    # get root element
    time_slot = None
    label = None
    root = xmlTree.getroot()
    for item in root.findall('./TIME_ORDER/'):
        time_slot_dict = item.attrib
        time_slot_id = time_slot_dict['TIME_SLOT_ID']
        time_value = time_slot_dict['TIME_VALUE']

        if frame <= int(time_value):
            time_slot = time_slot_id
            # print(time_slot_id)
            break

    for label in root.findall('./TIER/'):
        for alignable_annotation in label.findall('./ALIGNABLE_ANNOTATION'):
            if time_slot == alignable_annotation.attrib['TIME_SLOT_REF2']:
                # print(alignable_annotation.attrib['TIME_SLOT_REF2'])
                # print(time_slot)
                for annotation_value in alignable_annotation.findall('./ANNOTATION_VALUE'):
                    # print(annotation_value.text)
                    label = annotation_value.text
                    return label

def correctFrameLabel(xmlTree):
    print(xmlTree)
    # get root element
    time_slot = None

    root = xmlTree.getroot()
    for item in root.findall('./TIME_ORDER/'):
        time_slot_dict = item.attrib
        time_slot_id = int(time_slot_dict['TIME_SLOT_ID'][2:])
        time_value = time_slot_dict['TIME_VALUE']
        print(int(time_value))

        if time_slot_id > 1 and time_slot_id % 2 == 1:
            time_slot_dict['TIME_VALUE'] = prev_time_value
        if time_slot_id % 2 == 0:
            prev_time_value = time_value


    xmlTree.write('./Annotations/red1_edited.eaf')
# correctFrameLabel(xmlTree)
