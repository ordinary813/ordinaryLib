package org.example;
import javax.persistence.*;

@Entity
@Table(name = "people")
public class Person
{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;
    private final String firstname;
    private final String lastname;
    @Column (name = "Password")
    private String password;

    @Column(name = "Email_address")
    private String email;

    @Transient
    private Car[] cars;

    @Column (name = "Number_of_cars")
    private int numberOfCars;

    public Person(String firstname, String lastname, String password, String email)
    {
        super();
        this.firstname = firstname;
        this.lastname = lastname;
        this.password = password;
        this.email = email;

        this.cars = new Car[]{};
        this.numberOfCars = 0;
    }

    public void addCar(Car newCar)
    {
        Car[] newCars = new Car[this.cars.length + 1];
        System.arraycopy(this.cars, 0, newCars, 0, this.cars.length);
        newCars[newCars.length - 1] = newCar;
        this.cars = newCars;
        this.numberOfCars++;
    }

    public void printPersonProperties()
    {
        System.out.println("Name: " + this.firstname + " " + this.lastname);
        System.out.println("Email: " + this.email);
        System.out.println("Password: " + this.password);
        System.out.println("Number of cars owned: " + this.numberOfCars);
        System.out.println();
    }

    public String getFirstname()
    {
        return this.firstname;
    }

    public String getLastname()
    {
        return this.lastname;
    }

    public String getPassword()
    {
        return this.password;
    }

    public String getEmail()
    {
        return this.email;
    }

    public int getId()
    {
        return this.id;
    }

    public void setPassword(String newPassword)
    {
        this.password = newPassword;
    }

    public void setEmail(String newEmail)
    {
        this.email = newEmail;
    }
}
