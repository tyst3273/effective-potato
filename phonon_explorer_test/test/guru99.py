import numpy
class FitFuncTemp:
    def func1(self,Energy,p1,p2,p3,p4):
        return (p1/p3)*numpy.exp(-(numpy.power(((Energy-p2)/p3),2)))+(p4/p3)*numpy.exp(-(numpy.power(((Energy-p2-100)/p3),2)))