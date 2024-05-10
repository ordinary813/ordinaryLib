package org.example;

import java.net.URL;
import java.util.ResourceBundle;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;

public class PrimaryController {

    @FXML
    private ResourceBundle resources;

    @FXML
    private URL location;

    @FXML
    private ComboBox<String> baseCB;

    @FXML
    private Button btn0;

    @FXML
    private Button btn1;

    @FXML
    private Button btn2;

    @FXML
    private Button btn3;

    @FXML
    private Button btn4;

    @FXML
    private Button btn5;

    @FXML
    private Button btn6;

    @FXML
    private Button btn7;

    @FXML
    private Button btn8;

    @FXML
    private Button btn9;

    @FXML
    private Button btnA;

    @FXML
    private Button btnAsterisk;

    @FXML
    private Button btnB;

    @FXML
    private Button btnC;

    @FXML
    private Button btnClear;

    @FXML
    private Button btnD;

    @FXML
    private Button btnE;

    @FXML
    private Button btnEqu;

    @FXML
    private Button btnF;

    @FXML
    private Button btnMinus;

    @FXML
    private Button btnPlus;

    @FXML
    private Button btnSlash;

    @FXML
    private GridPane grid;

    @FXML
    private TextField screen;

    private boolean errorFlag = false;
    private String currentBaseString = "DEC";
    private int currentBase = 10;

    @FXML
    void appendChar(ActionEvent event)
    {
        if(this.errorFlag) {
            this.screen.setText("");
            this.errorFlag=false;
        }

        Button sourceButton = (Button) event.getSource();
        String character = sourceButton.getText();
        String currentScreen = this.screen.getText();

        currentScreen = currentScreen + character;
        this.screen.setText(currentScreen);

    }

    @FXML
    void clearScreen(ActionEvent event) {
        this.screen.setText("");
    }

    @FXML
    void computeExpression(ActionEvent event) {
        String expression = this.screen.getText();
        String baseString = this.baseCB.getSelectionModel().getSelectedItem();
        int base = 0;

        switch (baseString) {
            case "BIN":
                base = 2;
                break;
            case "OCT":
                base = 8;
                break;
            case "DEC":
                base = 10;
                break;
            case "HEX":
                base = 16;
                break;
        }

        String res = ArithmeticApp.expressionResult(expression, base);
        // if the result is error, I want to remove the error message as soon as a new number is added
        if(res.startsWith("Error"))
            this.errorFlag = true;
        this.screen.setText(res);

    }

    private String convertToNewBase(String expression, int toBase)
    {
        //this gives the result in the current base
//        String result = ArithmeticApp.expressionResult(expression,this.currentBase);
        //retrieve the decimal value of the expression
        int decimalRes = Integer.parseInt(expression,this.currentBase);
        //convert it to the new base
        expression = ArithmeticApp.convertToBase(decimalRes,toBase);
        return String.valueOf(expression);
    }

    @FXML
    void chooseBase(ActionEvent event) {
        String chosen = this.baseCB.getSelectionModel().getSelectedItem();
        String currExpression = screen.getText();
        currExpression = ArithmeticApp.expressionResult(currExpression,this.currentBase);

        if(chosen.equals("BIN"))
        {
            this.btn2.setDisable(true);
            this.btn3.setDisable(true);
            this.btn4.setDisable(true);
            this.btn5.setDisable(true);
            this.btn6.setDisable(true);
            this.btn7.setDisable(true);
            this.btn8.setDisable(true);
            this.btn9.setDisable(true);
            this.btnA.setDisable(true);
            this.btnB.setDisable(true);
            this.btnC.setDisable(true);
            this.btnD.setDisable(true);
            this.btnE.setDisable(true);
            this.btnF.setDisable(true);

            if(!currExpression.startsWith("Error"))
                this.screen.setText(convertToNewBase(currExpression, 2));
            else {
                this.screen.setText(currExpression);
                this.errorFlag = true;
            }

            this.currentBase = 2;
            this.currentBaseString = "BIN";
        }

        if(chosen.equals("OCT"))
        {
            this.btn2.setDisable(false);
            this.btn3.setDisable(false);
            this.btn4.setDisable(false);
            this.btn5.setDisable(false);
            this.btn6.setDisable(false);
            this.btn7.setDisable(false);
            this.btn8.setDisable(true);
            this.btn9.setDisable(true);
            this.btnA.setDisable(true);
            this.btnB.setDisable(true);
            this.btnC.setDisable(true);
            this.btnD.setDisable(true);
            this.btnE.setDisable(true);
            this.btnF.setDisable(true);

            if(!currExpression.startsWith("Error"))
                this.screen.setText(convertToNewBase(currExpression, 8));
            else {
                this.screen.setText(currExpression);
                this.errorFlag = true;
            }

            this.currentBase = 8;
            this.currentBaseString = "OCT";
        }

        if(chosen.equals("DEC"))
        {
            this.btn2.setDisable(false);
            this.btn3.setDisable(false);
            this.btn4.setDisable(false);
            this.btn5.setDisable(false);
            this.btn6.setDisable(false);
            this.btn7.setDisable(false);
            this.btn8.setDisable(false);
            this.btn9.setDisable(false);
            this.btnA.setDisable(true);
            this.btnB.setDisable(true);
            this.btnC.setDisable(true);
            this.btnD.setDisable(true);
            this.btnE.setDisable(true);
            this.btnF.setDisable(true);

            if(!currExpression.startsWith("Error"))
                this.screen.setText(convertToNewBase(currExpression, 10));
            else {
                this.screen.setText(currExpression);
                this.errorFlag = true;
            }

            this.currentBase = 10;
            this.currentBaseString = "DEC";
        }

        if(chosen.equals("HEX"))
        {
            this.btn2.setDisable(false);
            this.btn3.setDisable(false);
            this.btn4.setDisable(false);
            this.btn5.setDisable(false);
            this.btn6.setDisable(false);
            this.btn7.setDisable(false);
            this.btn8.setDisable(false);
            this.btn9.setDisable(false);
            this.btnA.setDisable(false);
            this.btnB.setDisable(false);
            this.btnC.setDisable(false);
            this.btnD.setDisable(false);
            this.btnE.setDisable(false);
            this.btnF.setDisable(false);

            if(!currExpression.startsWith("Error"))
                this.screen.setText(convertToNewBase(currExpression, 16));
            else {
                this.screen.setText(currExpression);
                this.errorFlag = true;
            }

            this.currentBase = 16;
            this.currentBaseString = "HEX";
        }
    }

    @FXML
    void initialize() {
        this.baseCB.getItems().add("BIN");
        this.baseCB.getItems().add("OCT");
        this.baseCB.getItems().add("DEC");
        this.baseCB.getItems().add("HEX");

        // I've chosen decimal to be the initial base, hence disabling of hex digits
        this.baseCB.setValue(this.currentBaseString);
        this.btnA.setDisable(true);
        this.btnB.setDisable(true);
        this.btnC.setDisable(true);
        this.btnD.setDisable(true);
        this.btnE.setDisable(true);
        this.btnF.setDisable(true);
    }

}
