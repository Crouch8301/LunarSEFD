#!/usr/bin/env python
# coding: utf-8

import datetime
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skyfield.api import load
from skyfield.framelib import ecliptic_frame
from skyfield.framelib import galactic_frame
from skyfield.positionlib import build_position
import h5py
import healpy as hp
import os
import cv2
import sys
import moviepy
from moviepy.editor import VideoFileClip

def main():
    
    ############# INPUT VARIABLES #####################
    
    ###The below variables are the input arguments to the program. Change them to suit your needs.###
    
    # Time of Interest
    Year = 2025
    Month = 8
    Day = 16
    Timezone = 8       #Perth time is +0800
    Hour = 9           # Local Time
    Minute = 27

    #Some Input Satellite Variables

    Inc = 30           # Inclination above the lunar equtorial plane
    Alt = 300          # Orbit altitude in km above the moon surface
    L_ant = 5          #antenna length in meters
    Freq = 10          # Measurement freq in MHz - should match frequency of sky model map below

    # To place the satellite we need to have a reference position. This is a position at which the bottom of the tilted orbit is closest to the earth, and the satellite is at the bottom of it's orbit.

    Y_ref = 2023       # reference year
    M_ref = 5          # reference month
    D_ref = 11         # reference day of month
    H_ref = 12         # reference hour
    M_ref = 0          # reference minute
    S_ref = 0          # reference second
    Zone_ref = 8       # timezone of reference time (Perth is +8)
    
    
    # This program contains 5 Functions:
                # Plotter - Creates an animation of the orbits of the earth, moon and satellite
                # Healmap - Creates a single image of the ULSA map with moon blockage, representing the satellite view at one particular position
                # Healvid - Creates an animation of the ULSA map with moon blockage, simulating the changing view of an orbiting satellite. Note this will temporarily create 1 file for each frame of the video, these will be autimatically deleted at completion of function.
                # SingleSEFD - Calculates the SEFD at the reference time and prints/returns the maximum, minimum, and average single pixel values
                # Analyse - Repeatedly calls SingleSEFD function over a period of time, and returns similar data to above
    
    
    ############# Function Specific Inputs ###############
    
    #General (applies to multiple)#
    PlotMoon = True        # [True/False] Whether or not to include moon blockage in the calculation/map
    
    #Plotter
    objects = [1,1,0,1]    # Which objects are visible in animation ; [Earth, Moon, Satellite, Satellite Orbital Plane] ; 1 = ON, 0 = OFF
    centre = "E"           # Which object is the animation centred on; Earth = "E", Moon = "M", Satellite = "S"
    Plotter_frames = 471   # Number of frames in the Plotter animation
    Plotter_step = datetime.timedelta(days = 1) # Period of simulation time each frame represents. Can change "seconds" to [hours, minutes, days, milliseconds]
    
    #Healvid
    Healvid_frames = 138   # Number of frames in the Healvid animation
    Healvid_step = datetime.timedelta(seconds = 50) # Period of simulation time each frame represents. Can change "seconds" to [hours, minutes, days, milliseconds]
    PlotSEFD_v = True      # [True/False] Whether or not to include SEFD numbers on plot
    
    #Healmap
    PlotSEFD_m = True      # [True/False] Whether or not to include SEFD numbers on plot
    PlotMap = True         # [True/False] Whether or not to plot and save the map (Default file name "ULSA Map at 10 MHz")
    
    #Analyse
    Analyse_Frames = 138   # Number of timesteps to analyse SEFD (Default is for single orbit about moon)
    Analyse_Step = datetime.timedelta(minutes = 1)  # Period of simulation time each frame represents. Can change "seconds" to [hours, minutes, days, milliseconds]
    
    
    ############# END OF INPUT VARIABLES ##############
    
    # Don't Edit the below lines
    
    Time_ref = datetime.datetime(year = Y_ref, month = Month, day = D_ref, hour = H_ref, minute = M_ref, second = S_ref) - datetime.timedelta(hours=Zone_ref)
    Time_curr = datetime.datetime(Year, Month, Day,Hour,Minute,0) - datetime.timedelta(hours=Timezone) + datetime.timedelta(days=118*2)
    
    
    ############ Function Selection #################
    
    # Uncomment/Comment the functions you do/don't wish to use. More than one can be used at the same time.
    
#    Plotter(Time_ref, Time_curr, Inc, Alt, objects, centre, Plotter_frames, Plotter_step) 

    Healmap(Time_ref, Time_curr, Inc, Alt, PlotSEFD_m, Freq, L_ant, PlotMap, PlotMoon)
    
#    Healvid(Time_ref, Time_curr, Inc, Alt, Healvid_frames, Healvid_step, PlotSEFD_v, Freq, L_ant)
    
#    SingleSEFD(Time_ref, Time_curr, Inc, Alt, Freq, L_ant, PlotMoon, True)

#    Analyse(Time_ref, Time_curr, Inc, Alt, Freq, L_ant, Analyse_Frames, Analyse_Step)
    
    ########### END OF PROGRAM INPUT ##########
    
    ### NOTES ###
    
    #Some variables are not listed above and must be changed in the relevant programs if needed. These include:
        # - The input sky map file name (currently "exp10.0MHz_sky_map_with_absorption.hdf5")
        # - The input map NSIDE and coordinate system (currently 64 and galactic coordinates)
        # - Filenames for generated files
        # - The temperature of the moon (currently 0 degrees K)
    
    
def Analyse(Time_ref, Time_curr, Inc, Alt, Freq, L_ant, Analyse_Frames, Analyse_Step):
    #Analyse - Repeatedly calls SingleSEFD function over a period of time, and records the minimum, maximum and average pixel SEFD for each frame
    #Time_ref - reference time as a datetime variable
    #Time_curr - time of interest as a datetime variable
    #Inc - satellite orbit inclination
    #Alt - altitude of satellite orbit
    #Freq - measurement frequency
    #L_ant - antenna length
    #Analyse_Frames - Number of timesteps to analyse SEFD (Default is for single orbit about moon)
    #Analyse_Step - Period of simulation time each frame represents.
    #Returns nothing, but prints the minimum average SEFD, maximum average SEFD, minimum minumum SEFD and maximum maximum SEFD from across all frames.
    
    
    Results = np.empty([Analyse_Frames, 3])
    TimeStore = np.empty([Analyse_Frames])
    Time = Time_curr
    
    for i in range(Analyse_Frames):
        TimeStore[i] = np.datetime64(Time)
        Results[i,:] = SingleSEFD(Time_ref, Time, Inc, Alt, Freq, L_ant, True, False)
        Time += Analyse_Step
    
    MinAvgInd = np.argmin(Results[:,2])
    MaxAvgInd = np.argmax(Results[:,2])
    MinMinInd = np.argmin(Results[:,0])
    MaxMaxInd = np.argmax(Results[:,1])
    
    print(f"Minimum Avg. SEFD: {Results[MinAvgInd, 2]} at: {np.array([TimeStore[MinAvgInd]/1000000], dtype='datetime64[s]')}")
    print(f"Maximum Avg. SEFD: {Results[MaxAvgInd, 2]} at: {np.array([TimeStore[MaxAvgInd]/1000000], dtype='datetime64[s]')}")
    print(f"Minimum Min. SEFD: {Results[MinMinInd, 0]} at: {np.array([TimeStore[MinMinInd]/1000000], dtype='datetime64[s]')}")
    print(f"Maximum Max. SEFD: {Results[MaxMaxInd, 1]} at: {np.array([TimeStore[MaxMaxInd]/1000000], dtype='datetime64[s]')}")
    
    
def SingleSEFD(Time_ref, Time_curr, Inc, Alt, Freq, L_ant, PlotMoon, Print):
    #SingleSEFD - computes the SEFD at the time of interest and prints the summarised data
    #Time_ref - reference time as a datetime variable
    #Time_curr - time of interest as a datetime variable
    #Inc - satellite orbit inclination
    #Alt - altitude of satellite orbit
    #Freq - measurement frequency
    #L_ant - antenna length
    #PlotMoon - [True/False] Whether or not to include moon blockage in the calculation/map
    #Print - [True/False] Whether or not to print the calculated values to the console
    #Returns the minimum, maximum and average single pixel SEFD. Also prints it to the console if Print=True
    
    Map,SEFDs = Healmap(Time_ref, Time_curr, Inc, Alt, False, Freq, L_ant, False, PlotMoon)
    
    if Print == True:
        print(f"Minimum SEFD: {np.min(SEFDs)}")
        print(f"Maximum SEFD: {np.max(SEFDs)}")
        print(f"Average SEFD: {np.mean(SEFDs)}")
    
    return(np.array([np.min(SEFDs), np.max(SEFDs), np.mean(SEFDs)]))
    
def SEFD(data, Freq, L_ant):
    #SEFD - computes the SEFD of the given input map
    #data - the input map, must be NSIDE 64 and in galactic coordinates
    #Freq - measurement frequency
    #L_ant - antenna length
    #Returns the SEFD array, with the SEFD corresponding to each pixel of the input map.
    
    
    NSIDE = 64
    angles = hp.pix2ang(NSIDE,np.arange(0,hp.nside2npix(NSIDE))) #Calculating position angles (standard spherical coordinates) for each pixel in the map
    pixelarea = hp.nside2pixarea(NSIDE, degrees=False) #Angular area of each pixel
    
    #Normalised Power Patterns
    PnX = (np.cos(angles[0])**2)*(np.cos(angles[1])**2) + (np.sin(angles[1])**2)
    PnY = (np.cos(angles[0])**2)*(np.sin(angles[1])**2) + (np.cos(angles[1])**2)
    PnZ = np.sin(angles[0])**2
    
    #Solid angles (should all be equal for identical antennas)
    SA_X = np.sum(PnX*pixelarea)
    SA_Y = np.sum(PnY*pixelarea)
    SA_Z = np.sum(PnZ*pixelarea)

    #Antenna Temperatures
    TaX = (1/SA_X)*np.sum(data*PnX*pixelarea)
    TaY = (1/SA_Y)*np.sum(data*PnY*pixelarea)
    TaZ = (1/SA_Z)*np.sum(data*PnZ*pixelarea)
    
    #Setting up variables for SEFD calculation
    kb = 1.38065E-23 # Boltzmann Constant in J/K
    k_Jy = 1380.65 # Boltzmann Constant in Jansky's
    n0 = 120*np.pi # Free Space Impedence\
    L_eff = L_ant/2 # Effective antenna length for short dipole
    Lambda = (3*(10**8))/(Freq*(10**6)) # Measurement wavelength
    Rant = 80*(np.pi**2)*((L_eff/Lambda)**2) # Antenna Resistance
    Vrms=3*np.sqrt(2)*1E-9 # RMS noise voltage
    Trx=(Vrms**2)/(4*kb*Rant) #Receiver Noise
    
    SEFDs = np.zeros(hp.nside2npix(64)) #Empty array to hold SEFD Data
    
    for n in range(hp.nside2npix(64)):
        #Calculates SEFD for each pixel and writes it to SEFDs array
        
        #Jones Matrix Components
        q_xTh = np.cos(angles[0][n])*np.cos(angles[1][n])
        q_xPh = -np.sin(angles[1][n])
        q_yTh = np.cos(angles[0][n])*np.sin(angles[1][n])
        q_yPh = np.cos(angles[1][n])
        q_zTh = -np.sin(angles[0][n])
        q_zPh = 0
        
        #Jones matrices and arrangements
        Q = np.array([[q_xTh, q_xPh],[q_yTh, q_yPh],[q_zTh, q_zPh]])
        J = Q*L_eff
        L1 = np.matmul(np.linalg.inv(np.matmul(J.conj().T, J)),J.conj().T)
        L2 = L1
        M = np.matmul(L2.conj().T,L1)
        
        #Ps = np.matmul(Q, Q.T)
        #SEFDs[n] = ((8*np.pi*k_Jy)/(3*Lambda**2))*np.sqrt(np.matmul(np.matmul(T_sys1.T,Ps*Ps),T_sys2))
        
        #System Temperatures
        T_sys1 = np.array([[TaX],[TaY],[TaZ]])+Trx
        T_sys2 = np.array([[TaX],[TaY],[TaZ]])+Trx
        
        #SEFD Calculation
        SEFDs[n] = ((4*k_Jy*Rant)/n0)*np.sqrt(np.matmul(np.matmul(T_sys1.T,M*M.conj()),T_sys2))
    
    
    return(SEFDs)
    
def Healvid(Time_ref, Time_curr, Inc, Alt, Healvid_frames, Healvid_step, PlotSEFD, Freq, L_ant):
    #Healvid - creates and saves a video of the satellite view using the ULSA map and halpix package
    #Time_ref - reference time as a datetime variable
    #Time_curr - time of interest as a datetime variable
    #Inc - satellite orbit inclination
    #Alt - altitude of satellite orbit
    #Healvid_frames - Number of frames in video
    #Healvid_step - Length of simulation time each frame represents
    #Does not return anything, but saves a GIF video with the name "Healpy Animation.gif".
    #Will temporarily create png file for each frame of the video, which are deleted automatically at the end of the program
    
    f=h5py.File("exp10.0MHz_sky_map_with_absorption.hdf5","r") #Input sky map, 10 MHz ULSA by default
    dat = f.get('data')                             #Map data field
    smooth = f.get('smooth_absorb')                 # Unused map data
    spectral = f.get('spectral_index')              # Unused map data

    map_data = np.array(dat)                        # Data Conversion

    f.close()
    
    frames = Healvid_frames                         #Number of frames for the video
    frame_step = Healvid_step                       #Time period that each frame represents
    
    filenames = []
    
    for i in range(frames):
        # Creates each frame of th video and saves them as png files
        
        Data = Positions(Time_ref, Time_curr, Inc, Alt) # Getting position of the earth, moon and satellite
        All_Pos = Data[0]                           # Grabbing just the positions
    
        Block_vec_ecl = All_Pos[1] - All_Pos[2]     # Vector between satellite and moon
    
        J2000 = datetime.datetime(2000,1,1,0,0,0)   #J2000 Epoch
        T_Delta = Time_curr - J2000                 # Time since J2000
        d = T_Delta.total_seconds()/(60*60*24) +1   # Time in days
        ecl = 23.4393 - (3.563**(-7))*d             #Obliquity of the ecliptic - Earth's tilt

        Block_vec_eq = [Block_vec_ecl[0], (Block_vec_ecl[1] * math.cos(math.radians(ecl)) - Block_vec_ecl[2] * math.sin(math.radians(ecl))), (Block_vec_ecl[1] * math.sin(math.radians(ecl)) + Block_vec_ecl[2] * math.cos(math.radians(ecl)))]
                                                    #Converting blocking vector from ecliptic coords into equatorial


        #Converting time of interest from datetime into format for Skyfield package
        ts = load.timescale()
        time_curr2 = ts.utc(Time_curr.year, Time_curr.month, Time_curr.day, Time_curr.hour, Time_curr.minute, Time_curr.second)
                                                    

        # Converting Equatorial Coordinates of Blocking Vector into Galactic Coordinatees
        position = build_position(Block_vec_eq, t=time_curr2)
        lat, lon, distance = position.frame_latlon(galactic_frame)
        lat2 = math.radians((lat.degrees+90)%360)    #Galactic Latitude
        lon2 = math.radians(lon.degrees)             #Galactic Longitude
    
    
        # Placing the moon on the healpix map
        vec = hp.ang2vec(lat2, lon2)
        block_angle = math.degrees(math.asin(1737.4/(1737.4+Alt)))
        ipix_disc = hp.query_disc(nside=64, vec=vec, radius=math.radians(block_angle)) #Calculating map pixels being blocked by moon
        Map = np.array(map_data)
        Map[ipix_disc] = 1      #Overwriting map data with 1 (Very low value) for blocked pixels. This represents the temperature of the pixel
        
        #Calculating SEFD for view
        SEFDs = SEFD(Map, Freq, L_ant)
        
        # Plotting and saving the map with moon blockage
        fig1 = plt.figure(dpi=300)
        cmap = mpl.colormaps['jet']
        #With SEFD
        if PlotSEFD == True:
            hp.mollview(np.log10(Map), fig = fig1, title = "Satellite View (10MHz)" , cmap=cmap, min= 4.8, max=6.2, sub = 211, margins = (0,0.1,0,0.1))
            hp.mollview(SEFDs/1000000, fig = fig1, title = "SEFD Map" , cmap=cmap, format = '%.1f', min= 5.1, max=6.9, sub = 212, margins = (0,0.15,0,0.1))
            plt.figtext( 0.2, 0.01, f"Min: {np.min(SEFDs)/1000000:.3} MJy", fontsize=10, ha="center")
            plt.figtext( 0.5, 0.01,f"Avg: {np.mean(SEFDs)/1000000:.3} MJy", fontsize=10, ha="center")
            plt.figtext( 0.8, 0.01,f"Max: {np.max(SEFDs)/1000000:.3} MJy", fontsize=10, ha="center")
            plt.figtext( 0.5, 0.1,f"MJy", fontsize=10, ha="center")
            plt.figtext( 0.5, 0.55,f"log(K)", fontsize=10, ha="center")
        #Without SEFD
        else:
            hp.mollview(np.log10(Map), fig = fig1, title = "ULSA Map at 10 MHz" , cmap=cmap, min= 4.8, max=6.2)
    
        #Save Figure
        fn = f"Sprectrum_{i}.jpg"
        filenames.append(fn)
        plt.savefig(fn)
        plt.clf()
        plt.close(fig1)
        
        Time_curr = Time_curr + frame_step


    #Combine frames into a video file
    firstFrame = cv2.imread(filenames[0])
    dimensions = firstFrame.shape
    writer = cv2.VideoWriter("Healpy Animation.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), 20, (dimensions[1], dimensions[0]))
    
    for filename in filenames:
        image = cv2.imread(filename)
        writer.write(image)
    
    writer.release()
    
    # Removes the png images
    for filename in set(filenames):
        os.remove(filename)
    
    mp4_video = VideoFileClip("Healpy Animation.mp4")
    mp4_video.write_gif("Healpy Animation.gif")
    
    
    

def Healmap(Time_ref, Time_curr, Inc, Alt, PlotSEFD, Freq, L_ant, PlotMap, PlotMoon):
    #Healmap - Creates and saves a single image of the satellite view using the ULSA map, with moon blockage
    #Identical to Healvid, but creates a single image instead of a video.
    #Time_ref - reference time as a datetime variable
    #Time_curr - time of interest as a datetime variable
    #Inc - satellite orbit inclination
    #Alt - altitude of satellite orbit
    #PlotSEFD - Whether or not to include SEFD numbers on plot
    #PlotMap - [True/False] Whether the map should be plotted or not
    #PlotMoon - [True/False] Whether the moon should be included or not
    #Returns the map data with moon blockage and the SEFD's data; Also saves a png file with the name "10MHZ_Spectrum.png"
    
    f=h5py.File("exp10.0MHz_sky_map_with_absorption.hdf5","r") # Input sky map, 10 MHz ULSA by default
    dat = f.get('data')                             # Map data field
    smooth = f.get('smooth_absorb')                 # Unused map data
    spectral = f.get('spectral_index')              # Unused map data

    map_data = np.array(dat)                        # Data Conversion

    if PlotMoon == True:
        Data = Positions(Time_ref, Time_curr, Inc, Alt) # Getting position of the earth, moon and satellite
        All_Pos = Data[0]                               # Grabbing just the positions

        Block_vec_ecl = All_Pos[1] - All_Pos[2]         # Vector between satellite and moon
    
    
        J2000 = datetime.datetime(2000,1,1,0,0,0)       # J2000 Epoch
        T_Delta = Time_curr - J2000                     # Time since J2000
        d = T_Delta.total_seconds()/(60*60*24) +1       # Time in days
        ecl = 23.4393 - (3.563**(-7))*d                 # Obliquity of the ecliptic - Earth's tilt

        Block_vec_eq = [Block_vec_ecl[0], (Block_vec_ecl[1] * math.cos(math.radians(ecl)) - Block_vec_ecl[2] * math.sin(math.radians(ecl))), (Block_vec_ecl[1] * math.sin(math.radians(ecl)) + Block_vec_ecl[2] * math.cos(math.radians(ecl)))]
                                                    #Converting blocking vector from ecliptic coords into equatorial
    
    
        #Converting time of interest from datetime into format for Skyfield package
        ts = load.timescale()
        time_curr2 = ts.utc(Time_curr.year, Time_curr.month, Time_curr.day, Time_curr.hour, Time_curr.minute, Time_curr.second)

    
        # Converting Equatorial Coordinates of Blocking Vector into Galactic Coordinatees
        position = build_position(Block_vec_eq, t=time_curr2)
        lat, lon, distance = position.frame_latlon(galactic_frame)
        lat2 = math.radians((lat.degrees+90)%360)       # Galactic Latitude
        lon2 = math.radians(lon.degrees)                # Galactic Longitude
    
    
        # Placing the moon on the healpix map
        vec = hp.ang2vec(lat2, lon2)
        block_angle = math.degrees(math.asin(1737.4/(1737.4+Alt)))
        ipix_disc = hp.query_disc(nside=64, vec=vec, radius=math.radians(block_angle)) #Calculating map pixels being blocked by moon
        Map = np.array(map_data)
        Map[ipix_disc] = 0                              #Overwriting map data with 1 (Very low value) for blocked pixels. This represents the temperature of the pixel
    
    else:
        Map = np.array(map_data)

    
    #Calculating SEFD for view
    SEFDs = SEFD(Map, Freq, L_ant)

    # Plotting and saving the map with moon blockage
    if PlotMap == True:
        #With SEFD
        if PlotSEFD == True:
            fig2 = plt.figure(dpi=300)
            cmap = mpl.colormaps['jet']
            hp.mollview(np.log10(Map), fig = fig2, title = "Satellite View (10MHz)" , cmap=cmap, min= 4.8, max=6.2, sub = 211, margins = (0,0.1,0,0.1))
            hp.mollview(SEFDs/1000000, fig = fig2, title = "SEFD Map" , cmap=cmap, format = '%.1f', sub = 212, margins = (0,0.15,0,0.1))
            plt.figtext( 0.2, 0.01, f"Min: {np.min(SEFDs)/1000000:.3} MJy", fontsize=10, ha="center")
            plt.figtext( 0.5, 0.01,f"Avg: {np.mean(SEFDs)/1000000:.3} MJy", fontsize=10, ha="center")
            plt.figtext( 0.8, 0.01,f"Max: {np.max(SEFDs)/1000000:.3} MJy", fontsize=10, ha="center")
            plt.figtext( 0.5, 0.1,f"MJy", fontsize=10, ha="center")
            plt.figtext( 0.5, 0.55,f"log(K)", fontsize=10, ha="center")
            plt.savefig('10MHZ_Spectrum.png')
            plt.clf()
            plt.close(fig2)
        #Without SEFD
        else:
            fig2 = plt.figure(dpi=300)
            cmap = mpl.colormaps['jet']
            hp.mollview(np.log10(Map), fig = fig2, title = "ULSA Map at 10 MHz" , cmap=cmap, min= 4.8, max=6.2)
            plt.savefig('10MHZ_Spectrum.png')
            plt.clf()
            plt.close(fig2)

    return(Map, SEFDs)
    
    
    
    
    
    
def Plotter(Time_ref, Time_curr, Inc, Alt, objects, centre, Plotter_frames, Plotter_step):
    #Plotter - Creates orbital animation plots
    #Time_ref - reference time as a datetime variable
    #Time_curr - time of interest as a datetime variable
    #Inc - satellite orbit inclination
    #Alt - altitude of satellite orbit
    #objects - list of 4 booleans representing whether the [Earth, Moon, Satellite, Sat orbital Plane] is visible
    #centre - Which object is the animation centred on; Earth = "E", Moon = "M", Satellite = "S"
    #Plotter_frames - Number of frames in the Plotter animation
    #Plotter_step - Length of simulation time each frame represents    
    #Returns nothing, but saves a GIF file named "animation.gif"
    

    Data = Positions(Time_ref, Time_curr, Inc, Alt) # Getting position of the earth, moon and satellite
    All_Pos = Data[0]                               # Getting just the positions
    
    E_pos = All_Pos[0]                              # Earth Coordinates (Heliocentric Ecliptic Rectangular)
    M_pos = All_Pos[1]                              # Moon Coordinates (Heliocentric Ecliptic Rectangular)
    S_pos = All_Pos[2]                              # Earth Coordinates (Heliocentric Ecliptic Rectangular)

    # Making Spheres to represent the objects (Not to scale)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    x_e = 5*(4.2635E-5) * x + E_pos[0]
    y_e = 5*(4.2635E-5) * y + E_pos[1]
    z_e = 5*(4.2635E-5) * z + E_pos[2]
    
    x_m = 1*(1.1611E-5) * x + M_pos[0]
    y_m = 1*(1.1611E-5) * y + M_pos[1]
    z_m = 1*(1.1611E-5) * z + M_pos[2]
    
    x_s = 8*(6.6846E-7) * x + S_pos[0]
    y_s = 8*(6.6846E-7) * y + S_pos[1]
    z_s = 8*(6.6846E-7) * z + S_pos[2]
    
    
    # Making the orbital plane if enabled
    xx=0
    yy=0
    zz=0
    
    if objects[3]:
        point  = M_pos
        normal = Data[1]
        d = -point.dot(normal)
        xx, yy = np.meshgrid(np.arange(M_pos[0]-3E-4,M_pos[0]+3E-4, 5E-5), np.arange(M_pos[1]-3E-4,M_pos[1]+3E-4, 5E-5))
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    
    # Creating the plt figure
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    
    # Setting the axes limits so they track the object of interest
    
    if centre == "M":
        xmin = M_pos[0] - 0.0001
        xmax = M_pos[0] + 0.0001
        ymin = M_pos[1] - 0.0001
        ymax = M_pos[1] + 0.0001
        zmin = M_pos[2] - 0.0001
        zmax = M_pos[2] + 0.0001
    
    elif centre == "E":
        xmin = E_pos[0] - 0.003
        xmax = E_pos[0] + 0.003
        ymin = E_pos[1] - 0.003
        ymax = E_pos[1] + 0.003
        zmin = E_pos[2] - 0.0001
        zmax = E_pos[2] + 0.0001
    
    elif centre == "S":
        xmin = S_pos[0] - 0.0001
        xmax = S_pos[0] + 0.0001
        ymin = S_pos[1] - 0.0001
        ymax = S_pos[1] + 0.0001
        zmin = S_pos[2] - 0.0001
        zmax = S_pos[2] + 0.0001
    
    else:       #Moon centered as default
        xmin = M_pos[0] - 0.0001
        xmax = M_pos[0] + 0.0001
        ymin = M_pos[1] - 0.0001
        ymax = M_pos[1] + 0.0001
        zmin = M_pos[2] - 0.0001
        zmax = M_pos[2] + 0.0001
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    
    
    #Plotting the objects
    Earth = [0]
    Plane = [0]
    Moon = [0]
    Satellite = [0]
    
    if objects[1]:
        Moon =  [ax.plot_surface(x_m, y_m, z_m, color = 'grey')]
    
    if objects[2]:
        Satellite = [ax.plot_surface(x_s, y_s, z_s, color = 'm')]
    
    if objects[0]:
        Earth = [ax.plot_surface(x_e, y_e, z_e, color = 'b')]
    
    if objects[3]:
        Plane = [ax.plot_surface(xx, yy, zz, color = 'r', alpha = 0.6)]
    
    
    #Set an equal aspect ratio
    ax.set_aspect('equal')
    
    #Animation 
    ani = animation.FuncAnimation(fig=fig, func = PlotUpdate, frames=Plotter_frames, interval=50, blit = False, fargs = (Time_ref, Time_curr, Inc, Alt, Earth, Moon, Satellite, Plane, ax, objects, centre, Plotter_step))
    ani.save('animation.gif')
    plt.show()


    
def PlotUpdate(frame, Time_ref, Time_curr, Inc, Alt, Earth, Moon, Satellite, Plane, ax, objects, centre, Plotter_step):
    #PlotUpdate - function used by funcanimation in Plotter function to update the plot for the animation
    #frame - frame number of animation
    #Time_ref - reference time as a datetime variable
    #Time_curr - time of interest as a datetime variable
    #Inc - satellite orbit inclination
    #Alt - altitude of satellite orbit
    #Earth - Earth plot object, to be updated
    #Moon - Moon plot object, to be updated
    #Satellite - Satellite plot object, to be updated
    #Plane - Plane plot object, to be updated
    #ax - figure axes
    #objects - list of 4 booleans representing whether the [Earth, Moon, Satellite, Sat orbital Plane] is visible
    #centre - Which object is the animation centred on; Earth = "E", Moon = "M", Satellite = "S"
    #Plotter_step - Length of simulation time each frame represents
    #Returns the Axes plot objects
    
    #Deleting the current data in the plot objects
    
    if objects[1]:
        Moon[0].remove()
    
    if objects[2]:
        Satellite[0].remove()
    
    if objects[0]:
        Earth[0].remove()
    
    if objects[3]:
        Plane[0].remove()    
    
    
    #Calculating the time of interest for the current frame
    Frame_time = Time_curr + Plotter_step * frame
    
    # Getting the positions of the objects
    Data = Positions(Time_ref, Frame_time, Inc, Alt)  # Getting position of the earth, moon and satellite
    AllPos = Data[0]                                  # Getting just the positions
    
    E_pos = AllPos[0]                                 # Earth Coordinates (Heliocentric Ecliptic Rectangular)
    M_pos = AllPos[1]                                 # Moon Coordinates (Heliocentric Ecliptic Rectangular)
    S_pos = AllPos[2]                                 # Earth Coordinates (Heliocentric Ecliptic Rectangular)
    

    # Making Spheres to represent the objects (Not to scale)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    x_e = 5*(4.2635E-5) * x + E_pos[0]
    y_e = 5*(4.2635E-5) * y + E_pos[1]
    z_e = 5*(4.2635E-5) * z + E_pos[2]
    
    x_m = 1*(1.1611E-5) * x + M_pos[0]
    y_m = 1*(1.1611E-5) * y + M_pos[1]
    z_m = 1*(1.1611E-5) * z + M_pos[2]
    
    x_s = 8*(6.6846E-7) * x + S_pos[0]
    y_s = 8*(6.6846E-7) * y + S_pos[1]
    z_s = 8*(6.6846E-7) * z + S_pos[2]
    
    
    # Making the orbital plane if enabled
    xx=0
    yy=0
    zz=0
    
    if objects[3]:
        point  = M_pos
        normal = Data[1]
        d = -point.dot(normal)
        xx, yy = np.meshgrid(np.arange(M_pos[0]-3E-4,M_pos[0]+3E-4, 5E-5), np.arange(M_pos[1]-3E-4,M_pos[1]+3E-4, 5E-5))
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    
    # Setting the axes limits so they track the object of interest
    
    if centre == "M":
        xmin = M_pos[0] - 0.0001
        xmax = M_pos[0] + 0.0001
        ymin = M_pos[1] - 0.0001
        ymax = M_pos[1] + 0.0001
        zmin = M_pos[2] - 0.0001
        zmax = M_pos[2] + 0.0001
    
    elif centre == "E":
        xmin = E_pos[0] - 0.003
        xmax = E_pos[0] + 0.003
        ymin = E_pos[1] - 0.003
        ymax = E_pos[1] + 0.003
        zmin = E_pos[2] - 0.0001
        zmax = E_pos[2] + 0.0001
    
    elif centre == "S":
        xmin = S_pos[0] - 0.0001
        xmax = S_pos[0] + 0.0001
        ymin = S_pos[1] - 0.0001
        ymax = S_pos[1] + 0.0001
        zmin = S_pos[2] - 0.0001
        zmax = S_pos[2] + 0.0001
    
    else:       #Moon centered as default
        xmin = M_pos[0] - 0.0001
        xmax = M_pos[0] + 0.0001
        ymin = M_pos[1] - 0.0001
        ymax = M_pos[1] + 0.0001
        zmin = M_pos[2] - 0.0001
        zmax = M_pos[2] + 0.0001
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    
    
    #Plotting the objects
    
    if objects[1]:
        Moon[0] = ax.plot_surface(x_m, y_m, z_m, color = 'grey')
    
    if objects[2]:
        Satellite[0] = ax.plot_surface(x_s, y_s, z_s, color = 'm')
    
    if objects[0]:
        Earth[0] = ax.plot_surface(x_e, y_e, z_e, color = 'b')
        
    if objects[3]:
        Plane[0] = ax.plot_surface(xx, yy, zz, color = 'r', alpha = 0.6)
    
    return(Earth, Moon, Satellite, Plane)






def Positions(time_ref, time_curr, inc, alt):
    # Placing the Satellite Array
    # The array is considere below as a single point (representing the centre of the array), can look into differedd views accross the array later
    # The below assumes a circular orbit, inclined at 30 degrees above the lunar equatorial plane, which itself is ~1.5 degrees above ecliptic plane
    # Orbital height is initially set at 300 km
    # Source information 1: https://doi.org/10.3847/1538-3881/aac6c6
    # Source information 2; Chen et al (DSL paper)
    
    #Time_ref - reference time as a datetime variable
    #Time_curr - time of interest as a datetime variable
    #Inc - satellite orbit inclination
    #Alt - altitude of satellite orbit
    #Returns the positions of the Earth, moon and satellite in heliocentric ecliptic rectangular coordinates; and the satellite orbital plane normal unit vector
    
    
    R_orbit = (1737+alt)/149597870.69 # Orbit radius in AU units
    
    
    #Converting times into format for Skyfield Package
    ts = load.timescale()
    time_curr2 = ts.utc(time_curr.year, time_curr.month, time_curr.day, time_curr.hour, time_curr.minute, time_curr.second)
    time_ref2 = ts.utc(time_ref.year, time_ref.month, time_ref.day, time_ref.hour, time_ref.minute, time_ref.second)

    
    #Getting positions of the planets using the Skyfield Package
    planets = load('de421.bsp')                       # ephemeris DE421
    
    Earth = planets['Earth']
    Moon = planets['Moon']
    Sun = planets['Sun']
    
    Plan_ref = np.array([Sun.at(time_ref2).observe(Earth).frame_xyz(ecliptic_frame).au, Sun.at(time_ref2).observe(Moon).frame_xyz(ecliptic_frame).au])
    Plan_curr = np.array([Sun.at(time_curr2).observe(Earth).frame_xyz(ecliptic_frame).au, Sun.at(time_curr2).observe(Moon).frame_xyz(ecliptic_frame).au])
    #Plan_ref - planet positions at reference time ; Plan_curr - planet positions at time of interest ; Both in heliocentric ecliptic rectangular coordinates
    
    
    # Calculating Satellite orbit period
    T_Delta = time_curr - time_ref
    T_sec = T_Delta.total_seconds()
    Period_sat = math.sqrt((4*(math.pi**2)*(((1737+alt)*1000)**3))/((6.673E-11)*(7.3477E22)))
    
    
    #Calculating Satellite orbital position and nodal precession
    Prec_period = (2*math.pi)/(1.5*((1738100**2)/((alt*1000+1738100)**2))*(202.7E-6)*((2*math.pi)/Period_sat)*math.cos(math.radians(inc)))
    
    Curr_orb_perc = (T_sec%Period_sat)/Period_sat          # Current orbit completion fraction
    Curr_orb_deg = 360*Curr_orb_perc                       # Current orbit completion expressed as an angle (0-360 degrees)
    Orb_prec = ((T_sec/Prec_period)*360)%360              # Orbital precession (nodal regression)
    
    # Uncomment below to print out the obital period and precession period. Note that this will print every time the Positions function runs (Once every frame for animations)
#    print(f"Satellite Orbital Period is {Period_sat/(60*60):0.2f} Hours")
#    print(f"Satellite Orbital Precession Period is {Prec_period/(60*60*24*365.25):0.2f} Years")
                
                
    # Rotation of orbital plane
    
    ME_vect_ref = Plan_ref[0] - Plan_ref[1]                # vector from moon to earth at reference time
    
    Orb_norm_ref = np.array([ME_vect_ref[0], ME_vect_ref[1], math.sqrt(ME_vect_ref[0]**2 + ME_vect_ref[1]**2) * math.tan(math.radians(60))]) 
                                                           # normal vector to reference satellite orbital plane
 
    Orb_norm_ref_mag = np.linalg.norm(ME_vect_ref)         # Magnitude of reference orbital normal vector
    
    Orb_unit_ref = Orb_norm_ref/Orb_norm_ref_mag           # ref orbital normal vector as a unit vector
    
    theta_ref = math.degrees(math.atan(Orb_unit_ref[1]/Orb_unit_ref[0])) # Starting angle on x-y plane of satellite orbit plane normal vector. Angle taken from positive x axis
    theta_curr = theta_ref + Orb_prec                      # current angle of satellit orbit plane normal vector
    
    xy_mag = math.sqrt(Orb_unit_ref[0]**2 + Orb_unit_ref[1]**2) # magnitude of sat plane norm x-y vectors, which is constant throughout precession
    
    Orb_unit_curr = np.array([xy_mag*math.cos(math.radians(theta_curr)), xy_mag*math.sin(math.radians(theta_curr)), Orb_unit_ref[2]]) 
                                                           # unit vector of current orbital plane normal
    
    
    
    # Finding satellite position based on orbital plane and orbit position
    
    Sat_pos = Pos_Solver(x0 = Plan_curr[1,0], y0 = Plan_curr[1,1], z0 = Plan_curr[1,2],x1 = Orb_unit_curr[0], y1 = Orb_unit_curr[1], z1 = Orb_unit_curr[2], R = R_orbit, Theta = inc , Beta = Curr_orb_deg )
    
    All_positions = np.array([[Plan_curr[0,0],Plan_curr[0,1],Plan_curr[0,2]], [Plan_curr[1,0],Plan_curr[1,1],Plan_curr[1,2]], [Sat_pos[0],Sat_pos[1],Sat_pos[2] ]])
    
#    print(Curr_orb_deg)
#    print(All_positions)
    
    return [All_positions, Orb_unit_curr] 

def Pos_Solver(x0,y0,z0,x1,y1,z1,R,Theta,Beta):
    # Pos_Solver - Solves for the position of the Satellite, by solving the intersection of a sphere (sat orbit) and a plane (orbital plane)
    # x0,y0,z0 - the centre of the moon heliocentric coordinates
    # x1,y1,z1 - the normal vector to the satellite orbital plane
    # R - the orbit radius, in AU units
    # Theta - the satellite orbit inclination angle to ecliptic plane (default 30 deg) (degrees)
    # Beta - the orbit completion angle (angle between the sat and the reference position) (0-360 degrees)
    # Returns the position of the satellite in heliocentric ecliptic rectangular coordinates

    # Solving for the z-coordinate, giving the third equation to solve for the 3 unknowns
    H = -R * math.cos(math.radians(Beta)) * math.sin(math.radians(Theta))  # Z-offset from moon centre
    z = z0 - H                                                             # z coordinate of the satellite, can be used to solve for other coordinates
    
    # Grouping Constant terms
    a = (x1**2 / y1**2)+1
    c = ((z1**2 / y1**2)+1) * ((z-z0)**2)
    b = ((2*x1*z1)/(y1**2))*(z-z0)
    
    # Checks if the discriminant of the quadratic formula is negative, which can occur as it gets really close to 0 due to small rounding errors and the likes
    if ((b-2*a*x0)**2 - 4*a*(c - R**2 + a*(x0**2) - b*x0)) < 0:
        xpos = -(b-2*a*x0)/(2*a)
        xneg = -(b-2*a*x0)/(2*a)
    else:
        # Solving quadratic formula
        xpos = (-(b-2*a*x0) + math.sqrt((b-2*a*x0)**2 - 4*a*(c - R**2 + a*(x0**2) - b*x0)))/(2*a)
        xneg = (-(b-2*a*x0) - math.sqrt((b-2*a*x0)**2 - 4*a*(c - R**2 + a*(x0**2) - b*x0)))/(2*a)
        
    x = 0
    
    #Figuring out whether to use the positive or negativ value (breaking the symmetry)
    if Beta <= 180 and y1 >= 0:
        x = xneg
    elif Beta > 180 and y1 >= 0:
        x = xpos
    elif Beta <= 180 and y1 < 0:
        x = xpos
    elif Beta > 180 and y1 < 0:
        x = xneg
    
    y = y0 - (x1*(x-x0) + z1*(z-z0))/y1  # Calculating the y position
    
    Position = np.array([x,y,z])         # Position of the satellite
    
    return Position
    

    
main()





