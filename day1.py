# function to add two numbers
def add_numbers(a, b):
    return a + b
# function to subtract two numbers
def subtract_numbers(a, b):
    return a - b
# function to multiply two numbers
def multiply_numbers(a, b):
    return a * b
# function to divide two numbers
def divide_numbers(a, b):
    if b == 0:
        return "Error: Division by zero not allowed"
    return a / b    

def calculator():
    print("Select operation:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")

    while True:
        # take input from the user
        choice = input("Enter choice (1/2/3/4): ")

        # check if choice is one of the four options
        if choice in ['1', '2', '3', '4']:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))
            
            if choice == '1':
                print(f"{num1} + {num2} = {add_numbers(num1, num2)}")
            elif choice == '2':
                print(f"{num1} - {num2} = {subtract_numbers(num1, num2)}")
            elif choice == '3':
                print(f"{num1} * {num2} = {multiply_numbers(num1, num2)}")
            elif choice == '4':
                print(f"{num1} / {num2} = {divide_numbers(num1, num2)}")
            # check if user wants another calculation
        next_calculation = input("Do you want to perform another calculation? (yes/no): ")
        if next_calculation.lower() != 'yes':
            break
# call the calculator function
calculator()