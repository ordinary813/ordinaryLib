package org.example;

import javax.persistence.*;

@Entity
@Table(name = "garages")
public class Garage
{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    @Column (name = "Address")
    private String address;

    @Column (name = "Phone_Number")
    private String phoneNumber;

    @Transient
    private Person[] owners;
    @Transient
    private Car[] cars;

    public Garage(Car[] cars, Person[] owners, String phoneNumber, String address)
    {
        this.cars = cars;
        this.owners = owners;
        this.phoneNumber = phoneNumber;
        this.address = address;
    }

    public Garage(String phoneNumber, String address)
    {
        this.cars = new Car[]{};
        this.owners = new Person[]{};
        this.phoneNumber = phoneNumber;
        this.address = address;
    }

    public void addCar(Car newCar)
    {
        Car[] newCars = new Car[this.cars.length + 1];
        System.arraycopy(this.cars, 0, newCars, 0, this.cars.length);
        newCars[newCars.length - 1] = newCar;
        this.cars = newCars;
    }

    public void addOwner(Person newOwner)
    {
        Person[] newOwners = new Person[this.owners.length + 1];
        System.arraycopy(this.owners, 0, newOwners, 0, this.owners.length);
        newOwners[newOwners.length - 1] = newOwner;
        this.owners = newOwners;
    }

    public Car[] getCars()
    {
        return cars;
    }

    public String getAddress()
    {
        return address;
    }

    public String getPhoneNumber()
    {
        return phoneNumber;
    }

    public Person[] getOwners()
    {
        return owners;
    }
}
