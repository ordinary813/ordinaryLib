package org.example;

import java.util.Scanner;

public class ArithmeticApp
{
    // Iterative method to compute the expression according to base
    public static String expressionResult(String expression, int base)
    {
        expression = expression.replaceAll("\\s+", "");

        // split the expression into numbers array, and operators array
        // (operators array first cell is always a blank string)
        String[] numbers = expression.split("[-+*/]");
        String[] operators = expression.split("[0-9A-F]+");

        if (numbers.length != operators.length && !(numbers.length == 1 && operators.length == 0))
            return "Error: invalid expression: \"\"";



        /* --------------------- Formatting Check --------------------- */
        // Consecutive operators case
        for (int i = 1; i < operators.length; i++)
        {
            if (operators[i].length() > 1)
            {
                return "Error: invalid expression: \"\"";
            }
        }

        if(expression.startsWith("/") || expression.startsWith("+") || expression.startsWith("*"))
            return "Error: invalid expression: \"\"";

        // Invalid character case
        for (char c : expression.toCharArray())
        {
            // base > 10 the expression should hold either digits, abcdef (technically only for hexadecimal), or operators
            if (base > 10)
            {
                if (!Character.isDigit(c) && "+-*/".indexOf(c) == -1 && "ABCDEF".indexOf(c) == -1)
                {
                    return "Error: invalid expression: \"\"";
                }
            } else // in base < 10 anything that is not a digit or an operator should not appear in the expression
            {
                if (!Character.isDigit(c) && "+-*/".indexOf(c) == -1)
                {
                    return "Error: invalid expression: \"\"";
                } else if (Character.isDigit(c))
                {
                    // if base is < 10, check if the current DIGIT is higher than the base
                    if (Integer.parseInt(String.valueOf(c), 10) >= base)
                    {
                        // Check if any digit is bigger than the base
                        return "Error: invalid expression: \"\"";
                    }
                }
            }
        }

        // Starts with a negative number case
        // when the first number is negative - change the first operator from "-" to "", and the first number to its negative number
        if (expression.charAt(0) == '-')
        {
            numbers = removeElement(numbers, 0);
            numbers[0] = '-' + numbers[0];

            operators = removeElement(operators, 0);
            operators = insertElement(operators, "", 0);
        }

        /* ------------------------- Operation Implementation ------------------------ */
        /*  Iterating over the operators array while computing each single operation   */
        /*  and shrinking the number array until there is only one number - the result */

        // iterate over all multiplications and divisions in the expression
        for (int i = 1; i < operators.length; i++)
        {
            if (operators[i].equals("*") || operators[i].equals("/"))
            {
                String currentComputation = "";
                // compute the current multiplication, else compute division
                if (operators[i].equals("*"))
                    currentComputation = convertToBase(Integer.parseInt(numbers[i - 1], base) * Integer.parseInt(numbers[i], base), base);

                if (operators[i].equals("/"))
                {
                    if (Integer.parseInt(numbers[i], base) == 0)
                    {
                        return "Error: trying to divide by 0 (evaluated:\"" + numbers[i] + "\")";
                    }
                    currentComputation = convertToBase(Integer.parseInt(numbers[i - 1], base) / Integer.parseInt(numbers[i], base), base);
                }

                // remove the current operator from the array
                operators = removeElement(operators, i);
                // remove the 2 computed numbers
                numbers = removeElement(numbers, i);
                numbers = removeElement(numbers, i - 1);

                // insert the result into the correct spot in the number array
                numbers = insertElement(numbers, currentComputation, i - 1);
                i--;
            }
        }

        // iterate over all subtractions and additions in the expression
        for (int i = 1; i < operators.length; i++)
        {
            String currentComputation = "";
            if (operators[i].equals("-"))
                currentComputation = convertToBase(Integer.parseInt(numbers[i - 1], base) - Integer.parseInt(numbers[i], base), base);

            if (operators[i].equals("+"))
                currentComputation = convertToBase(Integer.parseInt(numbers[i - 1], base) + Integer.parseInt(numbers[i], base), base);

            operators = removeElement(operators, i);
            numbers = removeElement(numbers, i);
            numbers = removeElement(numbers, i - 1);

            numbers = insertElement(numbers, currentComputation, i - 1);
            i--;
        }
        // in the end, there should be one number in the numbers array - the result
        return numbers[0];
    }

    // remove element from index in the String array
    public static String[] removeElement(String[] arr, int index)
    {
        String[] newArr = new String[arr.length - 1];

        int newIndex = 0;
        for (int i = 0; i <= newArr.length; i++)
        {
            if (i != index)
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
        for (int i = 0; i < newArr.length; i++)
        {
            if (i == index)
            {
                newArr[i] = toInsert;
                i++;
            }
            if (i < newArr.length)
                newArr[i] = arr[oldIndex];
            oldIndex++;
        }
        return newArr;
    }

    // recieves a decimal number and a base and converts it into the string in base
    public static String convertToBase(int number, int base)
    {
        StringBuilder sb = new StringBuilder();
        boolean neg = false;

        // for string manipulation purposes, we will divide the
        // positive number and add a "-" in the first cell if necessary
        if (number < 0)
        {
            number = -number;
            neg = true;
        }
        // append the current remainder into the string to return, until we cannot divide the number anymore
        do
        {
            int remainder = number % base;
            if (remainder < 10)
            {
                sb.insert(0, remainder);
            } else
            {
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