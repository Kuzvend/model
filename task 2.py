import numpy
import requests
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
from scipy import special, constants
from numpy import arange, abs, sum
from matplotlib import pyplot as plt


# читаем XML файл
data = requests.get('https://jenyay.net/uploads/Student/Modelling/task_02.xml').text
root = ET.fromstring(data)
for element in root.iter('variant'):
    if element.get('number') == '14':
        D = float(element.get('D'))
        f_min = float(element.get('fmin'))
        f_max = float(element.get('fmax'))

# задаем константы для рассчетов
n_end = 10
f_step = 100000
r = D / 2
f_arange = arange(f_min, f_max, f_step)
wavelength_arange = constants.c / f_arange
k_arange = 2 * constants.pi / wavelength_arange


# h
def f4(n, x):
    return special.spherical_jn(n, x) + 1j * special.spherical_yn(n, x)


# b
def f3(n, x):
    return (x * special.spherical_jn(n - 1, x) - n * special.spherical_jn(n, x)) / (x * f4(n - 1, x) - n * f4(n, x))


# a
def f2(n, x):
    return special.spherical_jn(n, x) / f4(n, x)


# ЭПР
rcs_arange = (wavelength_arange ** 2) / numpy.pi * (abs(sum([((-1) ** n) * (n+0.5) * (f3(n, k_arange * r) - f2(n, k_arange * r)) for n in range(1, n_end)], axis=0)) ** 2)

if not os.path.isdir('result'):
    os.mkdir('result')
else:
    print('Уже есть такая директрория')

counter = 0

# Создаем корневой элемент xml документа с атрибутом version и encoding
root = ET.Element("data")

# Создаем дочерние элементы frequencydata, lambdadata и rcsdata
frequencydata = ET.SubElement(root, "frequencydata")
lambdadata = ET.SubElement(root, "lambdadata")
rcsdata = ET.SubElement(root, "rcsdata")



for f, lambda1, rcs in zip(f_arange, wavelength_arange, rcs_arange):
    ET.SubElement(frequencydata, "f").text = str(f) + ' Гц'
    ET.SubElement(lambdadata, "lambda").text = str(lambda1) + ' м'
    ET.SubElement(rcsdata, "rcs").text = str(rcs) + ' м^2'

tree = ET.ElementTree(root)

# Преобразуем XML в строку с отступами и переводами строк
xml_string = ET.tostring(root, encoding="utf-8").decode("utf-8")
dom = xml.dom.minidom.parseString(xml_string)
formatted_xml = dom.toprettyxml(indent="  ")  # два пробела для отступ

with open('result/data.xml', 'w') as file:

    file.write(formatted_xml)


# Строим график
plt.plot(f_arange, rcs_arange)
plt.show()

