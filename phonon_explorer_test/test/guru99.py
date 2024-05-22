import numpy
class FitFuncTemp:
    def func1(self,Energy,p1,p2,p3,p4,p5,p6,p7,p8):
        return (p1/p3)*numpy.exp(-(numpy.power(((Energy-p2)/p3),2)))+(p4/p3)*numpy.exp(-(numpy.power(((Energy-p2-100)/p3),2)))+(p5/p3)*numpy.exp(-(numpy.power(((Energy-p2-200)/p3),2)))+(p6/p3)*numpy.exp(-(numpy.power(((Energy-p2-300)/p3),2)))+(p7/p3)*numpy.exp(-(numpy.power(((Energy-p2-400)/p3),2)))+(p8/p3)*numpy.exp(-(numpy.power(((Energy-p2-500)/p3),2)))