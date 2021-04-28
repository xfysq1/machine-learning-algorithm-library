import math
import sys
import xlrd
import xlwt
import numpy as np
import pandas as pd

data = xlrd.open_workbook('F:\DDDATA\lxy0e.xlsx')
table = data.sheet_by_index(0)
nrows = table.nrows
ncols = table.ncols













        values = []

        for col in range(s.ncols):

            values.append(s.cell(row,col).value)

        print values

        x_data1.append(values[0])

        y_data1.append(values[1])