package org.example;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.scene.layout.AnchorPane;

import java.io.IOException;

public class App extends Application {

    private static Scene scene;

    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(App.class.getResource("primary.fxml"));
        Parent root = fxmlLoader.load();

        //get the root node of anchorpane
        AnchorPane anchorPane = (AnchorPane) root;

        // get width and height of anchorpane
        double width = anchorPane.getPrefWidth();
        double height = anchorPane.getPrefHeight();

        scene = new Scene(loadFXML(), width, height);
        stage.setScene(scene);
        stage.setTitle("Calculator");
        stage.show();
    }

    static void setRoot() throws IOException {
        scene.setRoot(loadFXML());
    }

    private static Parent loadFXML() throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(App.class.getResource("primary" + ".fxml"));
        return fxmlLoader.load();
    }

    public static void main(String[] args) {
        launch();
    }

}