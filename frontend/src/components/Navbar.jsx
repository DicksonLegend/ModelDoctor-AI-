import React from 'react';
import { NavLink } from 'react-router-dom';

export default function Navbar() {
  return (
    <nav className="navbar" id="main-navbar">
      <NavLink to="/" className="navbar-brand">
        🩺 <span className="gradient-text">ModelDoctor</span> AI+
      </NavLink>
      <ul className="navbar-links">
        <li>
          <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''} end>
            Dashboard
          </NavLink>
        </li>
        <li>
          <NavLink to="/results" className={({ isActive }) => isActive ? 'active' : ''}>
            Results
          </NavLink>
        </li>
        <li>
          <NavLink to="/compare" className={({ isActive }) => isActive ? 'active' : ''}>
            Compare
          </NavLink>
        </li>
        <li>
          <NavLink to="/monitor" className={({ isActive }) => isActive ? 'active' : ''}>
            Monitor
          </NavLink>
        </li>
      </ul>
    </nav>
  );
}
