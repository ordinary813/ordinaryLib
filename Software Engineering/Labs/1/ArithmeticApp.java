import java.util.Scanner;

public class ArithmeticApp
{
    public static void main(String[] args)
    {
        Scanner myObj = new Scanner(System.in);
        String base;
        int baseInt;

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
                // base is not a valid base
                System.out.print("Error – this base isn’t supported. ");
            }
        }
        System.out.println("Enter expression:");
        String expression = myObj.nextLine();

        String result = expressionResult(expression, baseInt);

        System.out.println("The value of expression " +expression+ " is : " +result);

        myObj.close();
    }

    public static String expressionResult(String expression, int base)
    {
        expression = expression.replaceAll("\\s+","");

        String[] numbers = expression.split("[-+*/]");
        String[] operators = expression.split("[0-9A-Fa-f]+");

        /* --------------------- Formatting Check --------------------- */
        // Check for consecutive operators
        for (int i = 1; i < operators.length; i++)
        {
            if (operators[i].length() > 1)
            {
                System.out.println("Error: invalid expression: \"\"");
                System.exit(1);
            }
        }

        // Check for invalid characters
        for (char c : expression.toCharArray())
        {
            if(base > 10)
            {
                if (!Character.isDigit(c) && "+-*/".indexOf(c) == -1 && "ABCDEFabcdef".indexOf(c) == -1)
                {
                    System.out.println("Error: Invalid character '" + c + "' in expression.");
                    System.exit(1);
                }
            } else
            {
                if (!Character.isDigit(c) && "+-*/".indexOf(c) == -1)
                {
                    System.out.println("Error: Invalid character '" + c + "' in expression.");
                    System.exit(1);
                } else if (Character.isDigit(c))
                {
                    if (Integer.parseInt(String.valueOf(c), 10) >= base)
                    {
                        // Check if any digit is bigger than the base
                        System.out.println("Error: invalid expression: \"\"");
                        System.exit(1);
                    }
                }
            }
        }

        // when the first number is negative - change the first operator to "", and the first number to the negative number
        if(expression.charAt(0) == '-')
        {
            numbers = removeElement(0, numbers);
            numbers[0] = '-' + numbers[0];

            operators = removeElement(0, operators);
            operators = insertElement(operators, "", 0);
        }

        /* ------------------------- Operation Implementation ------------------------ */
        // iterate over all multiplications in the expression
        for(int i = 1; i < operators.length; i++)
        {
            if(operators[i].equals("*"))
            {
                String currentComputation = convertToBase(Integer.parseInt(numbers[i-1], base) * Integer.parseInt(numbers[i], base),base);

                operators = removeElement(i, operators);
                numbers = removeElement(i, numbers);
                numbers = removeElement(i-1, numbers);

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
                    System.out.println("Error: trying to divide by 0 (evaluated:" + numbers[i] + ")");
                    System.exit(1);
                }

                String currentComputation = convertToBase(Integer.parseInt(numbers[i-1], base) / Integer.parseInt(numbers[i], base),base);

                operators = removeElement(i, operators);
                numbers = removeElement(i, numbers);
                numbers = removeElement(i-1, numbers);

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

                operators = removeElement(i, operators);
                numbers = removeElement(i, numbers);
                numbers = removeElement(i-1, numbers);

                numbers = insertElement(numbers, currentComputation, i-1);
                i--;
            }

            if(operators[i].equals("+"))
            {
                String currentComputation = convertToBase(Integer.parseInt(numbers[i-1], base) + Integer.parseInt(numbers[i], base),base);

                operators = removeElement(i, operators);
                numbers = removeElement(i, numbers);
                numbers = removeElement(i-1, numbers);

                numbers = insertElement(numbers, currentComputation, i-1);
                i--;
            }
        }

        return numbers[0];
    }

    public static String[] removeElement(int index , String[] arr)
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
        // Convert the integer to the specified base
        StringBuilder sb = new StringBuilder();
        boolean neg = false;

        if(number < 0)
        {
            number = -number;
            neg = true;
        }
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
