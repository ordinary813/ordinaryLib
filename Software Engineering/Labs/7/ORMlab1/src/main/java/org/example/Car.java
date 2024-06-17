package org.example;

import javax.persistence.*;

@Entity
@Table(name = "cars")
public class Car
{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;
    private String licensePlate;
    private double price;
    @Column(name = "manufacturing_year")
    private int year;

    @ManyToOne(fetch = FetchType.LAZY)  // Adjust fetch type as per your requirement
    @JoinColumn(name = "owner_id")  // name of the foreign key column in the cars table
    private Person owner;
    private String ownerName;

    @Transient
    private Image img;

    @Transient
    private Garage[] availableGarages;

    public Car()
    {
    }

    public Car(String licensePlate, double price, int year, Person owner, String ownerName)
    {
        super();
        this.licensePlate = licensePlate;
        this.price = price;
        this.year = year;

        this.availableGarages = new Garage[]{};
        this.owner = owner;
        if(owner != null)
        {
            this.ownerName = ownerName;
        }
    }

    public void addGarage(Garage newGarage)
    {
        Garage[] newGarages = new Garage[this.availableGarages.length + 1];
        System.arraycopy(this.availableGarages, 0, newGarages, 0 , this.availableGarages.length);
        newGarages[newGarages.length - 1] = newGarage;
        this.availableGarages = newGarages;
    }

    public void printCarProperties()
    {
        System.out.println("Id: " +this.id);
        System.out.println("License plate: " + this.licensePlate);
        System.out.println("Price: " + this.price);
        System.out.println("Year: " + this.year);
        System.out.println();
    }

    public Person getOwner()
    {
        return owner;
    }

    public String getLicensePlate()
    {
        return licensePlate;
    }

    public Garage[] getAvailableGarages()
    {
        return availableGarages;
    }

    public void setLicensePlate(String licensePlate)
    {
        this.licensePlate = licensePlate;
    }

    public double getPrice()
    {
        return price;
    }

    public void setPrice(double price)
    {
        this.price = price;
    }

    public int getYear()
    {
        return year;
    }

    public void setYear(int year)
    {
        this.year = year;
    }

    public int getId()
    {
        return id;
    }

    public void setOwner(Person p)
    {
        this.owner = p;
    }
}