import React, { useState, useEffect, useCallback } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Button, 
  TextField, 
  Grid, 
  Paper, 
  CircularProgress,
  LinearProgress,
  Alert,
  Snackbar,
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Backdrop,
  Card,
  CardMedia,
  CardContent,
  IconButton,
  Skeleton,
  Fade,
  Grow
} from '@mui/material';
import { 
  Upload, 
  Delete, 
  CheckCircle, 
  Error,
  CloudUpload,
  Psychology,
  AutoAwesome,
  Refresh
} from '@mui/icons-material';
import Dropzone from 'react-dropzone';
import axios from 'axios';

// Configure axios defaults
axios.defaults.baseURL = process.env.REACT_APP_API_URL || '';

function App() {
  // State management
  const [uploadedImages, setUploadedImages] = useState([]);
  const [sessionID, setSessionID] = useState(null);
  const [taskID, setTaskID] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  
  // Loading states
  const [isUploading, setIsUploading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  
  // UI state
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });
  const [uploadProgress, setUploadProgress] = useState(0);
  
  // Generation parameters
  const [generationParams, setGenerationParams] = useState({
    num_inference_steps: 50,
    guidance_scale: 7.5,
    num_images: 1,
    negative_prompt: '',
    seed: null
  });
  
  // Training parameters
  const [trainingParams, setTrainingParams] = useState({
    model_name: 'stabilityai/stable-diffusion-xl-base-1.0',
    instance_prompt: 'a photo of sks person',
    num_train_steps: 1000,
    learning_rate: 5e-6
  });

  // Load available models on mount
  useEffect(() => {
    loadModels();
  }, []);

  // Poll training status when training
  useEffect(() => {
    if (!taskID || !isTraining) return;

    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`/training-status/${taskID}`);
        const status = response.data;
        
        setTrainingStatus(status.status);
        setTrainingProgress(status.progress * 100);
        
        if (status.status === 'completed' || status.status === 'failed') {
          setIsTraining(false);
          clearInterval(interval);
          
          if (status.status === 'completed') {
            showNotification('Training completed successfully!', 'success');
            loadModels(); // Reload models to include new one
          } else {
            showNotification(`Training failed: ${status.error}`, 'error');
          }
        }
      } catch (error) {
        console.error('Error fetching training status:', error);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [taskID, isTraining]);

  const loadModels = async () => {
    setIsLoadingModels(true);
    try {
      const response = await axios.get('/models');
      setModels(response.data.models);
      if (response.data.models.length > 0 && !selectedModel) {
        setSelectedModel(response.data.models[0].name);
      }
    } catch (error) {
      showNotification('Failed to load models', 'error');
    } finally {
      setIsLoadingModels(false);
    }
  };

  const showNotification = (message, severity = 'info') => {
    setNotification({ open: true, message, severity });
  };

  const handleUpload = async (acceptedFiles) => {
    setIsUploading(true);
    setUploadProgress(0);
    
    const formData = new FormData();
    acceptedFiles.forEach((file) => {
      formData.append('files', file);
    });

    try {
      const response = await axios.post('/upload-images/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        },
      });
      
      setSessionID(response.data.session_id);
      setUploadedImages(acceptedFiles.map(file => ({
        name: file.name,
        url: URL.createObjectURL(file)
      })));
      showNotification(`Uploaded ${response.data.count} images successfully`, 'success');
    } catch (error) {
      showNotification('Failed to upload images', 'error');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleTrain = async () => {
    if (!sessionID) return;
    
    setIsTraining(true);
    setTrainingProgress(0);
    
    const formData = new FormData();
    formData.append('session_id', sessionID);
    formData.append('model_name', trainingParams.model_name);
    formData.append('instance_prompt', trainingParams.instance_prompt);
    formData.append('num_train_steps', trainingParams.num_train_steps);
    formData.append('learning_rate', trainingParams.learning_rate);
    
    try {
      const response = await axios.post('/train-model/', formData);
      setTaskID(response.data.task_id);
      setTrainingStatus(response.data.status);
      showNotification('Training started', 'info');
    } catch (error) {
      setIsTraining(false);
      showNotification('Failed to start training', 'error');
    }
  };

  const handleGenerate = async () => {
    if (!prompt || !selectedModel) return;
    
    setIsGenerating(true);
    setGeneratedImages([]);
    
    try {
      const response = await axios.post('/generate/', {
        prompt: prompt,
        model_name: selectedModel,
        ...generationParams
      });
      
      const imageUrls = response.data.images.map(path => path);
      setGeneratedImages(imageUrls);
      showNotification(`Generated ${response.data.count} images`, 'success');
    } catch (error) {
      showNotification('Failed to generate images', 'error');
    } finally {
      setIsGenerating(false);
    }
  };

  const clearImages = () => {
    uploadedImages.forEach(img => URL.revokeObjectURL(img.url));
    setUploadedImages([]);
    setSessionID(null);
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        DreamBooth Studio
      </Typography>
      <Typography variant="subtitle1" align="center" color="text.secondary" gutterBottom>
        Train custom AI models and generate unique images
      </Typography>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        {/* Upload Section */}
        <Grid item xs={12} md={6}>
          <Grow in timeout={500}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Box display="flex" alignItems="center" mb={2}>
                <CloudUpload sx={{ mr: 1 }} />
                <Typography variant="h5">Upload Training Images</Typography>
              </Box>
              
              <Dropzone onDrop={handleUpload} disabled={isUploading}>
                {({ getRootProps, getInputProps, isDragActive }) => (
                  <Box
                    {...getRootProps()}
                    sx={{
                      border: '2px dashed',
                      borderColor: isDragActive ? 'primary.main' : 'grey.400',
                      borderRadius: 2,
                      p: 4,
                      textAlign: 'center',
                      cursor: 'pointer',
                      bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                      transition: 'all 0.3s ease',
                      mb: 2,
                      '&:hover': {
                        borderColor: 'primary.main',
                        bgcolor: 'action.hover'
                      }
                    }}
                  >
                    <input {...getInputProps()} />
                    <Upload fontSize="large" color="action" />
                    <Typography variant="body1" sx={{ mt: 1 }}>
                      {isDragActive
                        ? 'Drop the images here...'
                        : 'Drag & drop images here, or click to select'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Upload 5-20 images of the same subject
                    </Typography>
                  </Box>
                )}
              </Dropzone>

              {isUploading && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress variant="determinate" value={uploadProgress} />
                  <Typography variant="caption" align="center" display="block" sx={{ mt: 1 }}>
                    Uploading... {uploadProgress}%
                  </Typography>
                </Box>
              )}

              {uploadedImages.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="subtitle2">
                      {uploadedImages.length} images uploaded
                    </Typography>
                    <Button size="small" startIcon={<Delete />} onClick={clearImages}>
                      Clear
                    </Button>
                  </Box>
                  <Box display="flex" gap={1} flexWrap="wrap">
                    {uploadedImages.map((img, index) => (
                      <Chip
                        key={index}
                        label={img.name}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              )}
            </Paper>
          </Grow>
        </Grid>

        {/* Training Section */}
        <Grid item xs={12} md={6}>
          <Grow in timeout={700}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Box display="flex" alignItems="center" mb={2}>
                <Psychology sx={{ mr: 1 }} />
                <Typography variant="h5">Training Configuration</Typography>
              </Box>

              <TextField
                fullWidth
                label="Instance Prompt"
                variant="outlined"
                value={trainingParams.instance_prompt}
                onChange={(e) => setTrainingParams({...trainingParams, instance_prompt: e.target.value})}
                sx={{ mb: 2 }}
                helperText="e.g., 'a photo of sks person'"
              />

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Base Model</InputLabel>
                <Select
                  value={trainingParams.model_name}
                  label="Base Model"
                  onChange={(e) => setTrainingParams({...trainingParams, model_name: e.target.value})}
                >
                  <MenuItem value="stabilityai/stable-diffusion-xl-base-1.0">SDXL Base</MenuItem>
                  <MenuItem value="runwayml/stable-diffusion-v1-5">SD 1.5</MenuItem>
                  <MenuItem value="stabilityai/stable-diffusion-2-1">SD 2.1</MenuItem>
                </Select>
              </FormControl>

              <Typography gutterBottom>Training Steps: {trainingParams.num_train_steps}</Typography>
              <Slider
                value={trainingParams.num_train_steps}
                onChange={(e, v) => setTrainingParams({...trainingParams, num_train_steps: v})}
                min={100}
                max={2000}
                step={100}
                marks
                valueLabelDisplay="auto"
                sx={{ mb: 3 }}
              />

              <Button
                fullWidth
                variant="contained"
                size="large"
                onClick={handleTrain}
                disabled={!sessionID || isTraining}
                startIcon={isTraining ? <CircularProgress size={20} /> : <Psychology />}
              >
                {isTraining ? 'Training...' : 'Start Training'}
              </Button>

              {isTraining && (
                <Fade in>
                  <Box sx={{ mt: 3 }}>
                    <Box display="flex" justifyContent="space-between" mb={1}>
                      <Typography variant="body2">Training Progress</Typography>
                      <Typography variant="body2">{Math.round(trainingProgress)}%</Typography>
                    </Box>
                    <LinearProgress variant="determinate" value={trainingProgress} />
                    <Chip
                      label={trainingStatus}
                      size="small"
                      color={trainingStatus === 'running' ? 'primary' : 'default'}
                      sx={{ mt: 1 }}
                    />
                  </Box>
                </Fade>
              )}
            </Paper>
          </Grow>
        </Grid>

        {/* Generation Section */}
        <Grid item xs={12}>
          <Grow in timeout={900}>
            <Paper sx={{ p: 3 }}>
              <Box display="flex" alignItems="center" mb={3}>
                <AutoAwesome sx={{ mr: 1 }} />
                <Typography variant="h5">Generate Images</Typography>
              </Box>

              <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                  <TextField
                    fullWidth
                    label="Prompt"
                    variant="outlined"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    multiline
                    rows={2}
                    sx={{ mb: 2 }}
                    placeholder="Describe the image you want to generate..."
                  />

                  <TextField
                    fullWidth
                    label="Negative Prompt (Optional)"
                    variant="outlined"
                    value={generationParams.negative_prompt}
                    onChange={(e) => setGenerationParams({...generationParams, negative_prompt: e.target.value})}
                    sx={{ mb: 2 }}
                    placeholder="What to avoid in the image..."
                  />
                </Grid>

                <Grid item xs={12} md={4}>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Model</InputLabel>
                    <Select
                      value={selectedModel}
                      label="Model"
                      onChange={(e) => setSelectedModel(e.target.value)}
                      disabled={isLoadingModels}
                    >
                      {models.map((model) => (
                        <MenuItem key={model.name} value={model.name}>
                          <Box display="flex" alignItems="center" gap={1}>
                            {model.name}
                            {model.loaded && <Chip label="Loaded" size="small" color="success" />}
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  <Box sx={{ mb: 2 }}>
                    <Typography gutterBottom>Steps: {generationParams.num_inference_steps}</Typography>
                    <Slider
                      value={generationParams.num_inference_steps}
                      onChange={(e, v) => setGenerationParams({...generationParams, num_inference_steps: v})}
                      min={10}
                      max={100}
                      valueLabelDisplay="auto"
                    />
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Typography gutterBottom>Guidance Scale: {generationParams.guidance_scale}</Typography>
                    <Slider
                      value={generationParams.guidance_scale}
                      onChange={(e, v) => setGenerationParams({...generationParams, guidance_scale: v})}
                      min={1}
                      max={20}
                      step={0.5}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                </Grid>
              </Grid>

              <Button
                variant="contained"
                size="large"
                onClick={handleGenerate}
                disabled={!prompt || isGenerating || !selectedModel}
                startIcon={isGenerating ? <CircularProgress size={20} /> : <AutoAwesome />}
                sx={{ mt: 2 }}
              >
                {isGenerating ? 'Generating...' : 'Generate Images'}
              </Button>

              {/* Generated Images Gallery */}
              {(isGenerating || generatedImages.length > 0) && (
                <Box sx={{ mt: 4 }}>
                  <Typography variant="h6" gutterBottom>
                    Generated Images
                  </Typography>
                  <Grid container spacing={2}>
                    {isGenerating ? (
                      // Show skeletons while generating
                      Array.from({ length: generationParams.num_images }).map((_, index) => (
                        <Grid item xs={12} sm={6} md={4} key={index}>
                          <Skeleton variant="rectangular" height={300} sx={{ borderRadius: 1 }} />
                        </Grid>
                      ))
                    ) : (
                      // Show generated images
                      generatedImages.map((image, index) => (
                        <Grid item xs={12} sm={6} md={4} key={index}>
                          <Fade in timeout={500 + index * 200}>
                            <Card>
                              <CardMedia
                                component="img"
                                image={image}
                                alt={`Generated ${index + 1}`}
                                sx={{ height: 300, objectFit: 'cover' }}
                              />
                              <CardContent>
                                <Button
                                  fullWidth
                                  variant="outlined"
                                  href={image}
                                  download={`generated-${index + 1}.png`}
                                >
                                  Download
                                </Button>
                              </CardContent>
                            </Card>
                          </Fade>
                        </Grid>
                      ))
                    )}
                  </Grid>
                </Box>
              )}
            </Paper>
          </Grow>
        </Grid>
      </Grid>

      {/* Loading Backdrop */}
      <Backdrop
        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={isUploading || isTraining || isGenerating}
      >
        <Box textAlign="center">
          <CircularProgress color="inherit" size={60} />
          <Typography variant="h6" sx={{ mt: 2 }}>
            {isUploading && 'Uploading images...'}
            {isTraining && 'Training model...'}
            {isGenerating && 'Generating images...'}
          </Typography>
        </Box>
      </Backdrop>

      {/* Notification Snackbar */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={() => setNotification({ ...notification, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setNotification({ ...notification, open: false })}
          severity={notification.severity}
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default App;