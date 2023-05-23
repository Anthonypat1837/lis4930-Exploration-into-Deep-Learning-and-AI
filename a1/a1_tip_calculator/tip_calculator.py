
# Program Requirements

print("Tip Calculator")
 
# Developer: Anthony Patregnani

# Course: LIS4930

# Semester: Summer 2023

print("\nProgram Requirements:\n"
      + "1. Must use float data type for user input (except, \"Party Number\").\n"
      + "2. Must round calculations to two decimal places.\n"
      + "3. Must format currency with dollar sign, and two decimal places.")

print("\nUser Input:")

meal_cost = float(input("Cost of meal: "))
tax_percent = float(input("Tax percent: "))
tip_percent = float(input("Tip percent: "))
people_num = int(input("Party Number: "))

tax_amount = round(meal_cost * (tax_percent / 100), 2)
due_amount = round(meal_cost + tax_amount, 2)
tip_amount = round((due_amount) * (tip_percent / 100), 2)
total = round(meal_cost + tax_amount + tip_amount, 2)
split = round(total / people_num, 2)



print("\nProgram Output:")
print("Subtotal:\t", "${0:,.2f}".format(meal_cost))
print("Tax:\t\t", "${0:,.2f}".format(tax_amount))
print("Amount due:\t", "${0:,.2f}".format(due_amount))
print("Gratuity:\t", "${0:,.2f}".format(tip_amount))
print("Total:\t\t", "${0:,.2f}".format(total))
print("Split " + "(" + str(people_num) + "):\t", "${0:,.2f}".format(split))
 
