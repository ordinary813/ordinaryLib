import java.util.Scanner;

public class ArithmeticApp
{
    public static void main(String[] args)
    {
        Scanner myObj = new Scanner(System.in);
        String base;
        int baseInt;

        /* ______________Prompt user to enter base until the right base is entered_______________ */
        while(true)
        {
            System.out.println("Enter base (2/8/10/16):");
            base = myObj.nextLine();

            if (base.equals("2") || base.equals("8") || base.equals("10") || base.equals("16"))
            {
                // convert the type of base to int and break out of the loop
                baseInt = Integer.parseInt(base);
                break;
            } else {
                // base is not a valid base - continue prompting user
                System.out.print("Error – this base isn’t supported. ");
            }
        }
        /* ______________________________________________________________________________________ */


        System.out.println("Enter expression:");
        String expression = myObj.nextLine();

        String result = expressionResult(expression, baseInt);
        System.out.println("The value of expression " +expression+ " is : " +result);

        // closing Scanner object
        myObj.close();
    }

    // Iterative method to compute the expression according to base
    public static String expressionResult(String expression, int base)
    {
        expression = expression.replaceAll("\\s+","");

        // split the expression into numbers array, and operators array
        // (operators array first cell is always a blank string)
        String[] numbers = expression.split("[-+*/]");
        String[] operators = expression.split("[0-9A-Fa-f]+");

        /* --------------------- Formatting Check --------------------- */
        // Consecutive operators case
        for (int i = 1; i < operators.length; i++)
        {
            if (operators[i].length() > 1)
            {
                System.out.println("Error: invalid expression: \"\"");
                System.exit(1);
            }
        }

        // Invalid character case
        for (char c : expression.toCharArray())
        {
            // base > 10 the expression should hold either digits, abcdef (technically only for hexadecimal), or operators
            if(base > 10)
            {
                if (!Character.isDigit(c) && "+-*/".indexOf(c) == -1 && "ABCDEFabcdef".indexOf(c) == -1)
                {
                    System.out.println("Error: invalid expression: \"\"");
                    System.exit(1);
                }
            } else // in base < 10 anything that is not a digit or an operator should not appear in the expression
            {
                if (!Character.isDigit(c) && "+-*/".indexOf(c) == -1)
                {
                    System.out.println("Error: invalid expression: \"\"");
                    System.exit(1);
                } else if (Character.isDigit(c))
                {
                    // if base is < 10, check if the current DIGIT is higher than the base
                    if (Integer.parseInt(String.valueOf(c), 10) >= base)
                    {
                        // Check if any digit is bigger than the base
                        System.out.println("Error: invalid expression: \"\"");
                        System.exit(1);
                    }
                }
            }
        }

        // Starts with a negative number case
        // when the first number is negative - change the first operator from "-" to "", and the first number to its negative number
        if(expression.charAt(0) == '-')
        {
            numbers = removeElement(numbers, 0);
            numbers[0] = '-' + numbers[0];

            operators = removeElement(operators, 0);
            operators = insertElement(operators, "", 0);
        }

        /* ------------------------- Operation Implementation ------------------------ */
        /*  Iterating over the operators array while computing each single operation   */
        /*  and shrinking the number array until there is only one number - the result */
        // iterate over all multiplications in the expression
        for(int i = 1; i < operators.length; i++)
        {
            if(operators[i].equals("*"))
            {
                // compute the current multiplication
                String currentComputation = convertToBase(Integer.parseInt(numbers[i-1], base) * Integer.parseInt(numbers[i], base),base);

                // remove the current operator from the array
                operators = removeElement(operators, i);
                // remove the 2 computed numbers
                numbers = removeElement(numbers, i);
                numbers = removeElement(numbers, i-1);

                // insert the result into the correct spot in the number array
                numbers = insertElement(numbers, currentComputation, i-1);
                i--;
            }
        }

        // iterate over all divisions in the expression
        for(int i = 1; i < operators.length; i++)
        {
            if(operators[i].equals("/"))
            {
                if(Integer.parseInt(numbers[i], base) == 0)
                {
                    System.out.println("Error: trying to divide by 0 (evaluated:\"" + numbers[i] + "\")");
                    System.exit(1);
                }

                String currentComputation = convertToBase(Integer.parseInt(numbers[i-1], base) / Integer.parseInt(numbers[i], base),base);

                operators = removeElement(operators, i);
                numbers = removeElement(numbers, i);
                numbers = removeElement(numbers, i-1);

                numbers = insertElement(numbers, currentComputation, i-1);
                i--;
            }
        }

        // iterate over all subtractions and additions in the expression
        for(int i = 1; i < operators.length; i++)
        {
            if(operators[i].equals("-"))
            {
                String currentComputation = convertToBase(Integer.parseInt(numbers[i-1], base) - Integer.parseInt(numbers[i], base),base);

                operators = removeElement(operators, i);
                numbers = removeElement(numbers, i);
                numbers = removeElement(numbers, i-1);

                numbers = insertElement(numbers, currentComputation, i-1);
                i--;
            }

            if(operators[i].equals("+"))
            {
                String currentComputation = convertToBase(Integer.parseInt(numbers[i-1], base) + Integer.parseInt(numbers[i], base),base);

                operators = removeElement(operators, i);
                numbers = removeElement(numbers, i);
                numbers = removeElement(numbers, i-1);

                numbers = insertElement(numbers, currentComputation, i-1);
                i--;
            }
        }
        // in the end, there should be one number in the numbers array - the result
        return numbers[0];
    }

    // remove element from index in the String array
    public static String[] removeElement(String[] arr, int index)
    {
        String[] newArr = new String[arr.length - 1];

        int newIndex = 0;
        for(int i = 0; i <= newArr.length; i++)
        {
            if(i != index)
            {
                newArr[newIndex] = arr[i];
                newIndex++;
            }
        }
        return newArr;
    }

    // insert toInsert into index in arr
    public static String[] insertElement(String[] arr, String toInsert, int index)
    {
        String[] newArr = new String[arr.length + 1];

        int oldIndex = 0;
        for(int i = 0; i < newArr.length; i++)
        {
            if(i == index)
            {
                newArr[i] = toInsert;
                i++;
            }
            if(i < newArr.length)
                newArr[i] = arr[oldIndex];
            oldIndex++;
        }
        return newArr;
    }

    // recieves a decimal number and a base and converts it to the string in base
    public static String convertToBase(int number, int base)
    {
        StringBuilder sb = new StringBuilder();
        boolean neg = false;

        // for string manipulation purposes, we will divide the
        // positive number and add a "-" in the first cell if necessary
        if(number < 0)
        {
            number = -number;
            neg = true;
        }
        // append the current remainder into the string to return, until we cannot divide the number anymore
        do {
            int remainder = number % base;
            if (remainder < 10) {
                sb.insert(0, remainder);
            } else {
                char hexDigit = (char) ('A' + (remainder - 10));
                sb.insert(0, hexDigit);
            }
            number /= base;
        } while (number != 0);

        if (neg)
            sb.insert(0, '-');

        return sb.toString();
    }

}
