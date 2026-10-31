/**
 * CropSense 3D Scene - Three.js powered immersive background
 * Features: Particle field, floating geometric shapes, interactive 3D leaf, scroll-based animations
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

(() => {
  "use strict";

  const safeGsap = {
    from: (...args) => (window.gsap && window.gsap.from) ? window.gsap.from(...args) : null,
    to: (...args) => (window.gsap && window.gsap.to) ? window.gsap.to(...args) : null,
  };

  // ============================================
  // CONFIGURATION
  // ============================================
  const CONFIG = {
    particleCount: 1500,
    particleSize: 0.02,
    particleColors: [0x10b981, 0x059669, 0x34d399, 0x6ee7b7, 0xa7f3d0],
    shapeCount: 12,
    leafGeometryDetail: 64,
    animationSpeed: 0.3,
    mouseInfluence: 0.02,
    scrollInfluence: 0.5,
  };

  // ============================================
  // STATE
  // ============================================
  let scene, camera, renderer, controls;
  let particles, particleGeometry, particleMaterial;
  let shapes = [];
  let heroLeaf = null;
  let mouse = new THREE.Vector2(0, 0);
  let targetMouse = new THREE.Vector2(0, 0);
  let scrollY = 0;
  let time = 0;
  let rafId = null;
  let isInitialized = false;
  let heroCanvas = null;
  let heroScene, heroCamera, heroRenderer;
  let heroLeafMesh = null;
  let heroAnimationId = null;

  // ============================================
  // INITIALIZATION
  // ============================================
  function init() {
    if (isInitialized) return;

    // Check for reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
      initFallback();
      return;
    }

    // Wait for DOM
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initThree);
    } else {
      initThree();
    }
  }

  function initThree() {
    setupMainScene();
    setupHeroLeaf();
    setupEventListeners();
    animate();
    isInitialized = true;
  }

  function initFallback() {
    // Simple CSS-only fallback
    document.body.classList.add('no-3d');
    initHeroLeafCSS();
  }

  // ============================================
  // MAIN SCENE (Background)
  // ============================================
  function setupMainScene() {
    const container = document.getElementById('three-canvas-container');
    if (!container) return;

    // Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x080f0a, 0.0015);

    // Camera
    const aspect = window.innerWidth / window.innerHeight;
    camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 2000);
    camera.position.set(0, 0, 50);

    // Renderer
    renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x080f0a, 0);
    container.appendChild(renderer.domElement);

    // Controls (subtle)
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enablePan = false;
    controls.enableZoom = false;
    controls.minPolarAngle = Math.PI / 2 - 0.3;
    controls.maxPolarAngle = Math.PI / 2 + 0.3;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.2;

    // Create particles
    createParticles();
    createFloatingShapes();
    createAmbientLight();

    // Handle resize
    window.addEventListener('resize', onResize, { passive: true });
  }

  function createParticles() {
    particleGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(CONFIG.particleCount * 3);
    const colors = new Float32Array(CONFIG.particleCount * 3);
    const sizes = new Float32Array(CONFIG.particleCount);
    const velocities = new Float32Array(CONFIG.particleCount * 3);
    const phases = new Float32Array(CONFIG.particleCount);

    for (let i = 0; i < CONFIG.particleCount; i++) {
      const i3 = i * 3;

      // Spherical distribution
      const radius = 30 + Math.random() * 70;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = radius * Math.cos(phi);

      // Color
      const color = new THREE.Color(CONFIG.particleColors[Math.floor(Math.random() * CONFIG.particleColors.length)]);
      colors[i3] = color.r;
      colors[i3 + 1] = color.g;
      colors[i3 + 2] = color.b;

      // Size variation
      sizes[i] = 0.5 + Math.random() * 1.5;

      // Velocity for organic movement
      velocities[i3] = (Math.random() - 0.5) * 0.0005;
      velocities[i3 + 1] = (Math.random() - 0.5) * 0.0005;
      velocities[i3 + 2] = (Math.random() - 0.5) * 0.0005;

      // Phase for wave animation
      phases[i] = Math.random() * Math.PI * 2;
    }

    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    particleGeometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
    particleGeometry.setAttribute('phase', new THREE.BufferAttribute(phases, 1));

    // Particle material with custom shader
    particleMaterial = new THREE.PointsMaterial({
      size: CONFIG.particleSize,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      sizeAttenuation: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);
  }

  function createFloatingShapes() {
    const geometries = [
      new THREE.OctahedronGeometry(1, 0),
      new THREE.TetrahedronGeometry(1, 0),
      new THREE.IcosahedronGeometry(1, 0),
      new THREE.DodecahedronGeometry(1, 0),
    ];

    for (let i = 0; i < CONFIG.shapeCount; i++) {
      const geometry = geometries[Math.floor(Math.random() * geometries.length)];
      const material = new THREE.MeshPhysicalMaterial({
        color: CONFIG.particleColors[Math.floor(Math.random() * CONFIG.particleColors.length)],
        metalness: 0.3,
        roughness: 0.4,
        transparent: true,
        opacity: 0.15,
        transmission: 0.1,
        thickness: 0.5,
      });

      const mesh = new THREE.Mesh(geometry, material);

      // Random position in a shell around camera
      const radius = 40 + Math.random() * 40;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      mesh.position.set(
        radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.sin(phi) * Math.sin(theta),
        radius * Math.cos(phi)
      );

      // Random rotation and scale
      mesh.rotation.set(
        Math.random() * Math.PI,
        Math.random() * Math.PI,
        Math.random() * Math.PI
      );
      const scale = 0.5 + Math.random() * 2;
      mesh.scale.setScalar(scale);

      // Animation properties
      mesh.userData = {
        originalPosition: mesh.position.clone(),
        rotationSpeed: new THREE.Vector3(
          (Math.random() - 0.5) * 0.001,
          (Math.random() - 0.5) * 0.001,
          (Math.random() - 0.5) * 0.001
        ),
        floatOffset: Math.random() * Math.PI * 2,
        floatSpeed: 0.0005 + Math.random() * 0.001,
        floatAmplitude: 2 + Math.random() * 3,
      };

      shapes.push(mesh);
      scene.add(mesh);
    }
  }

  function createAmbientLight() {
    // Subtle ambient
    const ambient = new THREE.AmbientLight(0x10b981, 0.3);
    scene.add(ambient);

    // Directional light for shapes
    const dirLight = new THREE.DirectionalLight(0x34d399, 0.5);
    dirLight.position.set(50, 50, 50);
    scene.add(dirLight);

    // Point lights for glow
    const pointLight1 = new THREE.PointLight(0x10b981, 0.5, 100);
    pointLight1.position.set(-30, 20, 30);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0x059669, 0.3, 100);
    pointLight2.position.set(30, -20, -30);
    scene.add(pointLight2);
  }

  // ============================================
  // HERO 3D LEAF (Interactive)
  // ============================================
  function setupHeroLeaf() {
    heroCanvas = document.getElementById('hero-3d-leaf');
    if (!heroCanvas) return;

    // Hero scene
    heroScene = new THREE.Scene();

    // Hero camera - orthographic for consistent sizing
    const aspect = heroCanvas.clientWidth / heroCanvas.clientHeight;
    const frustumSize = 8;
    heroCamera = new THREE.OrthographicCamera(
      frustumSize * aspect / -2,
      frustumSize * aspect / 2,
      frustumSize / 2,
      frustumSize / -2,
      0.1,
      100
    );
    heroCamera.position.set(0, 0, 10);
    heroCamera.lookAt(0, 0, 0);

    // Hero renderer
    heroRenderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    });
    heroRenderer.setSize(heroCanvas.clientWidth, heroCanvas.clientHeight);
    heroRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    heroRenderer.setClearColor(0x000000, 0);
    heroCanvas.appendChild(heroRenderer.domElement);

    // Create leaf
    createHeroLeaf();

    // Hero lights
    const heroAmbient = new THREE.AmbientLight(0xffffff, 0.8);
    heroScene.add(heroAmbient);

    const heroDirLight = new THREE.DirectionalLight(0xffffff, 1);
    heroDirLight.position.set(5, 10, 7);
    heroScene.add(heroDirLight);

    const heroRimLight = new THREE.DirectionalLight(0x10b981, 0.5);
    heroRimLight.position.set(-5, -5, -5);
    heroScene.add(heroRimLight);

    // Hero interaction
    heroCanvas.addEventListener('mousemove', onHeroMouseMove);
    heroCanvas.addEventListener('mouseleave', onHeroMouseLeave);
    heroCanvas.addEventListener('touchmove', onHeroTouchMove, { passive: true });

    // Handle resize
    window.addEventListener('resize', onHeroResize);
  }

  function createHeroLeaf() {
    // Create a procedural leaf geometry
    const leafShape = createLeafShape();
    const leafGeometry = new THREE.ExtrudeGeometry(leafShape, {
      depth: 0.05,
      bevelEnabled: true,
      bevelThickness: 0.02,
      bevelSize: 0.02,
      bevelSegments: 4,
    });
    leafGeometry.center();

    // Custom shader material for leaf
    const leafMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x1a4d2e,
      metalness: 0,
      roughness: 0.6,
      transmission: 0.05,
      thickness: 0.1,
      clearcoat: 0.3,
      clearcoatRoughness: 0.2,
      side: THREE.DoubleSide,
    });

    heroLeafMesh = new THREE.Mesh(leafGeometry, leafMaterial);
    heroLeafMesh.rotation.x = -Math.PI / 2;
    heroLeafMesh.rotation.z = Math.PI / 6;
    heroScene.add(heroLeafMesh);

    // Add vein lines
    addLeafVeins();

    // Add deficiency spot (animated)
    addDeficiencySpot();

    // Entrance animation
    safeGsap.from(heroLeafMesh.scale, {
      x: 0, y: 0, z: 0,
      duration: 1.5,
      ease: 'elastic.out(1, 0.5)',
      delay: 0.3,
    });

    safeGsap.from(heroLeafMesh.rotation, {
      x: -Math.PI,
      duration: 2,
      ease: 'power3.out',
      delay: 0.5,
    });
  }

  function createLeafShape() {
    const shape = new THREE.Shape();
    const points = [
      [0, -4],    // Tip
      [1.5, -3],
      [2.5, -1.5],
      [3, 0],
      [3.2, 1.5],
      [3, 3],
      [2.5, 4],
      [1.5, 4.5],
      [0, 4.5],   // Base center
      [-1.5, 4.5],
      [-2.5, 4],
      [-3, 3],
      [-3.2, 1.5],
      [-3, 0],
      [-2.5, -1.5],
      [-1.5, -3],
      [0, -4],
    ];

    shape.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
      shape.lineTo(points[i][0], points[i][1]);
    }
    return shape;
  }

  function addLeafVeins() {
    const veinMaterial = new THREE.LineBasicMaterial({
      color: 0x0f3d1e,
      transparent: true,
      opacity: 0.4,
      linewidth: 1,
    });

    // Central vein
    const centralVein = new THREE.BufferGeometry();
    const centralPoints = [];
    for (let y = -3.8; y <= 4; y += 0.1) {
      centralPoints.push(0, y, 0.06);
    }
    centralVein.setAttribute('position', new THREE.Float32BufferAttribute(centralPoints, 3));
    const centralLine = new THREE.Line(centralVein, veinMaterial);
    heroLeafMesh.add(centralLine);

    // Side veins
    for (let i = 1; i <= 6; i++) {
      const y = -3 + i * 1.1;
      const width = 2.5 - i * 0.3;

      const veinGeometry = new THREE.BufferGeometry();
      const veinPoints = [];
      for (let x = -width; x <= width; x += 0.1) {
        const curve = Math.sin((x + width) / (width * 2) * Math.PI) * 0.3;
        veinPoints.push(x, y + curve, 0.06);
      }
      veinGeometry.setAttribute('position', new THREE.Float32BufferAttribute(veinPoints, 3));
      const vein = new THREE.Line(veinGeometry, veinMaterial);
      heroLeafMesh.add(vein);
    }
  }

  function addDeficiencySpot() {
    // Create a yellowish spot to indicate deficiency
    const spotGeometry = new THREE.CircleGeometry(0.6, 32);
    const spotMaterial = new THREE.MeshBasicMaterial({
      color: 0xf5a623,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
      depthTest: false,
    });
    const spot = new THREE.Mesh(spotGeometry, spotMaterial);
    spot.position.set(1.5, -1, 0.08);
    spot.rotation.x = -Math.PI / 2;
    heroLeafMesh.add(spot);
    heroLeafMesh.userData.deficiencySpot = spot;

    // Animate spot appearance
    safeGsap.to(spotMaterial, {
      opacity: 0.7,
      duration: 2,
      delay: 2,
      ease: 'power2.out',
      repeat: -1,
      yoyo: true,
      repeatDelay: 1,
    });

    // Pulse animation
    safeGsap.to(spot.scale, {
      x: 1.2, y: 1.2,
      duration: 1.5,
      ease: 'sine.inOut',
      repeat: -1,
      yoyo: true,
    });
  }

  function onHeroMouseMove(e) {
    const rect = heroCanvas.getBoundingClientRect();
    targetMouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    targetMouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  }

  function onHeroMouseLeave() {
    targetMouse.set(0, 0);
  }

  function onHeroTouchMove(e) {
    if (e.touches.length > 0) {
      const rect = heroCanvas.getBoundingClientRect();
      targetMouse.x = ((e.touches[0].clientX - rect.left) / rect.width) * 2 - 1;
      targetMouse.y = -((e.touches[0].clientY - rect.top) / rect.height) * 2 + 1;
    }
  }

  function onHeroResize() {
    if (!heroCanvas || !heroCamera || !heroRenderer) return;

    const width = heroCanvas.clientWidth;
    const height = heroCanvas.clientHeight;
    const aspect = width / height;
    const frustumSize = 8;

    heroCamera.left = frustumSize * aspect / -2;
    heroCamera.right = frustumSize * aspect / 2;
    heroCamera.top = frustumSize / 2;
    heroCamera.bottom = frustumSize / -2;
    heroCamera.updateProjectionMatrix();

    heroRenderer.setSize(width, height);
    heroRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  }

  // ============================================
  // CSS FALLBACK FOR HERO LEAF
  // ============================================
  function initHeroLeafCSS() {
    const container = document.getElementById('hero-3d-leaf');
    if (!container) return;

    container.innerHTML = `
      <div class="css-leaf-wrapper">
        <svg class="css-leaf" viewBox="0 0 200 250" aria-hidden="true">
          <defs>
            <linearGradient id="leafGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#1a4d2e"/>
              <stop offset="50%" stop-color="#10b981"/>
              <stop offset="100%" stop-color="#059669"/>
            </linearGradient>
            <filter id="leafGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="8" result="blur"/>
              <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          <path class="leaf-main" d="M100,10 C140,20 160,60 165,110 C170,160 165,200 145,225 C125,240 100,240 100,240 C100,240 75,240 55,225 C35,200 30,160 35,110 C40,60 60,20 100,10 Z" fill="url(#leafGradient)" filter="url(#leafGlow)"/>
          <path class="leaf-vein-main" d="M100,10 Q100,120 100,230" fill="none" stroke="#0f3d1e" stroke-width="1.5" opacity="0.4"/>
          <g class="leaf-veins" fill="none" stroke="#0f3d1e" stroke-width="0.8" opacity="0.3">
            <path d="M100,40 Q130,45 155,55"/>
            <path d="M100,40 Q70,45 45,55"/>
            <path d="M100,80 Q125,90 145,105"/>
            <path d="M100,80 Q75,90 55,105"/>
            <path d="M100,120 Q120,130 135,145"/>
            <path d="M100,120 Q80,130 65,145"/>
            <path d="M100,160 Q115,165 125,180"/>
            <path d="M100,160 Q85,165 75,180"/>
          </g>
          <ellipse class="deficiency-spot" cx="130" cy="60" rx="25" ry="18" fill="#f5a623" opacity="0.6" transform="rotate(-20 130 60)"/>
        </svg>
      </div>
    `;

    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
      .css-leaf-wrapper { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }
      .css-leaf { width: 100%; height: 100%; max-width: 380px; animation: leafFloat 6s ease-in-out infinite; }
      @keyframes leafFloat { 0%,100% { transform: translateY(0) rotate(-2deg); } 50% { transform: translateY(-15px) rotate(2deg); } }
      .deficiency-spot { animation: spotPulse 2s ease-in-out infinite; }
      @keyframes spotPulse { 0%,100% { opacity: 0.4; transform: scale(1); } 50% { opacity: 0.8; transform: scale(1.1); } }
    `;
    document.head.appendChild(style);
  }

  // ============================================
  // EVENT LISTENERS
  // ============================================
  function setupEventListeners() {
    // Mouse tracking for main scene
    document.addEventListener('mousemove', (e) => {
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    }, { passive: true });

    // Scroll tracking
    window.addEventListener('scroll', () => {
      scrollY = window.scrollY;
    }, { passive: true });

    // Intersection Observer for scroll animations
    setupScrollAnimations();
  }

  function setupScrollAnimations() {
    // Animate step cards
    const stepObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry, index) => {
        if (entry.isIntersecting) {
          safeGsap.from(entry.target, {
            y: 50,
            opacity: 0,
            duration: 0.8,
            delay: index * 0.15,
            ease: 'power3.out',
          });
          stepObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.2 });

    document.querySelectorAll('.step-3d').forEach(el => {
      stepObserver.observe(el);
    });
  }

  // ============================================
  // ANIMATION LOOP
  // ============================================
  function animate() {
    rafId = requestAnimationFrame(animate);
    time += 0.016;

    // Smooth mouse follow
    targetMouse.lerp(mouse, 0.05);

    // Update main scene
    updateMainScene();

    // Update hero leaf
    updateHeroLeaf();

    // Render
    if (renderer && scene && camera) {
      controls.update();
      renderer.render(scene, camera);
    }

    if (heroRenderer && heroScene && heroCamera) {
      heroRenderer.render(heroScene, heroCamera);
    }
  }

  function updateMainScene() {
    if (!particles || !particleGeometry) return;

    const positions = particleGeometry.attributes.position.array;
    const velocities = particleGeometry.attributes.velocity.array;
    const phases = particleGeometry.attributes.phase.array;
    const count = CONFIG.particleCount;

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;

      // Organic wave movement
      const wave = Math.sin(time * 0.5 + phases[i]) * 0.02;
      positions[i3] += velocities[i3] + wave * Math.cos(phases[i]);
      positions[i3 + 1] += velocities[i3 + 1] + wave * Math.sin(phases[i]);
      positions[i3 + 2] += velocities[i3 + 2];

      // Mouse influence
      positions[i3] += targetMouse.x * CONFIG.mouseInfluence * (1 - Math.abs(positions[i3]) / 100);
      positions[i3 + 1] += targetMouse.y * CONFIG.mouseInfluence * (1 - Math.abs(positions[i3 + 1]) / 100);

      // Scroll influence - particles move with scroll
      positions[i3 + 1] += scrollY * 0.0001 * Math.sin(phases[i]);

      // Boundary wrapping
      const dist = Math.sqrt(positions[i3]**2 + positions[i3+1]**2 + positions[i3+2]**2);
      if (dist > 120) {
        const radius = 30 + Math.random() * 40;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
        positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
        positions[i3 + 2] = radius * Math.cos(phi);
      }
    }

    particleGeometry.attributes.position.needsUpdate = true;

    // Rotate particle system slowly
    if (particles) {
      particles.rotation.y += 0.00005;
      particles.rotation.x += 0.00002;
    }

    // Update floating shapes
    shapes.forEach((mesh, i) => {
      const data = mesh.userData;

      // Rotation
      mesh.rotation.x += data.rotationSpeed.x;
      mesh.rotation.y += data.rotationSpeed.y;
      mesh.rotation.z += data.rotationSpeed.z;

      // Floating
      const floatY = Math.sin(time * data.floatSpeed + data.floatOffset) * data.floatAmplitude;
      mesh.position.y = data.originalPosition.y + floatY;

      // Subtle mouse attraction
      mesh.position.x += (targetMouse.x * 2 - mesh.position.x) * 0.0005;
      mesh.position.z += (targetMouse.y * 2 - mesh.position.z) * 0.0005;

      // Scroll parallax
      mesh.position.y -= scrollY * 0.001;
    });
  }

  function updateHeroLeaf() {
    if (!heroLeafMesh) return;

    // Smooth mouse follow rotation
    heroLeafMesh.rotation.y += (targetMouse.x * 0.3 - heroLeafMesh.rotation.y) * 0.05;
    heroLeafMesh.rotation.x += (-targetMouse.y * 0.2 - heroLeafMesh.rotation.x - Math.PI / 2) * 0.05;
    heroLeafMesh.rotation.z += (targetMouse.x * 0.1 - heroLeafMesh.rotation.z + Math.PI / 6) * 0.05;

    // Subtle breathing animation
    const breathe = 1 + Math.sin(time * 0.5) * 0.02;
    heroLeafMesh.scale.setScalar(breathe);

    // Animate deficiency spot position
    const spot = heroLeafMesh.userData.deficiencySpot;
    if (spot) {
      spot.position.x = 1.5 + Math.sin(time * 0.7) * 0.3;
      spot.position.y = -1 + Math.cos(time * 0.5) * 0.2;
    }
  }

  // ============================================
  // RESIZE HANDLERS
  // ============================================
  function onResize() {
    if (!renderer || !camera) return;

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  }

  // ============================================
  // CLEANUP
  // ============================================
  function dispose() {
    if (rafId) cancelAnimationFrame(rafId);

    if (renderer) {
      renderer.dispose();
      renderer.domElement.remove();
    }
    if (heroRenderer) {
      heroRenderer.dispose();
      heroRenderer.domElement.remove();
    }

    // Dispose geometries and materials
    if (particleGeometry) particleGeometry.dispose();
    if (particleMaterial) particleMaterial.dispose();

    shapes.forEach(mesh => {
      mesh.geometry.dispose();
      mesh.material.dispose();
    });

    if (heroLeafMesh) {
      heroLeafMesh.geometry.dispose();
      heroLeafMesh.material.dispose();
      heroLeafMesh.children.forEach(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
    }

    isInitialized = false;
  }

  // ============================================
  // PUBLIC API
  // ============================================
  window.CropSense3D = {
    init,
    dispose,
    getScene: () => scene,
    getHeroScene: () => heroScene,
  };

  // Auto-initialize
  init();
})();