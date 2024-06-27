import math

class Beta_distribution_online_learning():
    def __init__(self, a, b, file):
        self.a = a
        self.b = b
        self.file = file

    def forward(self):
        Read_file = self.file.readline()

        line = 1
        while Read_file:
            Count_One = Read_file.count('1')
            Count_Zero = Read_file.count('0')
            Total = Count_One + Count_Zero

            One_Prob = Count_One / Total
            Zero_Prob = Count_Zero / Total
            likelihood = math.factorial(Total) / math.factorial(Count_One) / math.factorial(Count_Zero) * One_Prob ** Count_One * Zero_Prob ** Count_Zero

            print("Line {}: {}".format(line, Read_file))
            print("Likelihood: {}".format(likelihood))
            print("Beta Prior: a={}, b={}".format(self.a, self.b))

            self.a += Count_One
            self.b += Count_Zero

            print("Beta Posterior: a={}, b={}".format(self.a, self.b))
            print()

            Read_file = self.file.readline()
            line += 1